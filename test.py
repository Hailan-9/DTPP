import yaml
import datetime
import torch
import argparse
import warnings
from tqdm import tqdm
from planner import Planner
from common_utils import *
from debug_utils import *
warnings.filterwarnings("ignore") 

from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
from nuplan.planning.simulation.main_callback.metric_summary_callback import MetricSummaryCallback
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
# NOTE nuplan控制器
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.nuboard.base.data_class import NuBoardFile
# TODO 参考了nuplan-devkit/nuplan/planning/script/run_nuboard.py中启动nuplan仿真的实现，其实就是相关函数的调用和传参

def build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir):
    """
    Builds the main experiment folder for simulation.
    :return: The main experiment folder path.
    """
    print('Building experiment folders...')

    exp_folder = pathlib.Path(output_dir)
    print(f'\nFolder where all results are stored: {exp_folder}\n')
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Build nuboard event file.
    nuboard_filename = exp_folder / (f'nuboard_{int(time.time())}' + NuBoardFile.extension())
    nuboard_file = NuBoardFile(
        simulation_main_path=str(exp_folder),
        simulation_folder=simulation_dir,
        metric_main_path=str(exp_folder),
        metric_folder=metric_dir,
        aggregator_metric_folder=aggregator_metric_dir,
    )

    metric_main_path = exp_folder / metric_dir
    metric_main_path.mkdir(parents=True, exist_ok=True)

    nuboard_file.save_nuboard_file(nuboard_filename)
    print('Building experiment folders...DONE!')

    return exp_folder.name


def build_simulation(experiment, planner, scenarios, output_dir, simulation_dir, metric_dir):
    runner_reports = []
    print(f'Building simulations from {len(scenarios)} scenarios...')
    # NOTE 自定义函数，初始化评估引擎，用于计算仿真指标。
    metric_engine = build_metrics_engine(experiment, output_dir, metric_dir)
    print('Building metric engines...DONE\n')

    # Iterate through scenarios
    for scenario in tqdm(scenarios, desc='Running simulation'):
        # Ego Controller and Perception
        # NOTE 使用回放控制器（LogPlaybackController）和轨迹观测（TracksObservation）。
        # NOTE LogPlaybackController: 一个控制器，用于回放预定义的轨迹（从场景日志中提取）。
        # NOTE TracksObservation: 感知模块，负责从场景中提取轨迹观测数据（如其他车辆的轨迹）。
        if experiment == 'open_loop_boxes':
            ego_controller = LogPlaybackController(scenario) 
            observations = TracksObservation(scenario)
        # NOTE closed_loop_nonreactive_agents: 使用 LQR 追踪器和运动学模型。
        # NOTE LQRTracker: 使用 LQR（线性二次调节器）算法实现的轨迹追踪器，负责计算车辆的控制输入。
        #     关键参数：
        #     q_longitudinal 和 r_longitudinal: 纵向误差和控制输入的权重。
        #     q_lateral 和 r_lateral: 横向误差和控制输入的权重。
        #     discretization_time: 离散化时间步长。
        #     tracking_horizon: 追踪的时间范围。
        #     jerk_penalty: 减少加加速度（jerk）的惩罚权重。
        #     curvature_rate_penalty: 曲率变化率的惩罚权重。
        #     stopping_proportional_gain 和 stopping_velocity: 停车控制参数。
        # KinematicBicycleModel: 运动学自行车模型，用于描述车辆的运动学行为。
        # get_pacifica_parameters: 自定义函数，获取车辆模型的参数（如车辆长度、宽度等）。
        # TwoStageController: 两阶段控制器，结合 LQR 追踪器和运动学模型
        elif experiment == 'closed_loop_nonreactive_agents':
            # TODO 控制器 使用LQR进行轨迹跟踪
            tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], 
                                 r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10, 
                                 jerk_penalty=1e-4, curvature_rate_penalty=1e-2, 
                                 stopping_proportional_gain=0.5, stopping_velocity=0.2)
            motion_model = KinematicBicycleModel(get_pacifica_parameters())
            ego_controller = TwoStageController(scenario, tracker, motion_model) 
            observations = TracksObservation(scenario)
            # NOTE closed_loop_reactive_agents: 同样使用 LQR 追踪器，但感知模块为 IDMAgents（基于智能驾驶模型的交互式感知）。
            # NOTE IDMAgents: 基于 IDM（智能驾驶模型）的感知模块，用于模拟交互式感知行为。
            #     关键参数：
            #     target_velocity: 目标速度。
            #     min_gap_to_lead_agent 和 headway_time: 与前车的最小间距和时间间隔。
            #     accel_max 和 decel_max: 最大加速度和减速度。
            #     open_loop_detections_types: 感知的目标类型（如行人、障碍物等）。
        elif experiment == 'closed_loop_reactive_agents':      
            tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], 
                                 r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10, 
                                 jerk_penalty=1e-4, curvature_rate_penalty=1e-2, 
                                 stopping_proportional_gain=0.5, stopping_velocity=0.2)
            motion_model = KinematicBicycleModel(get_pacifica_parameters())
            ego_controller = TwoStageController(scenario, tracker, motion_model) 
            observations = IDMAgents(target_velocity=10, min_gap_to_lead_agent=1.0, headway_time=1.5,
                                     accel_max=1.0, decel_max=2.0, scenario=scenario,
                                     open_loop_detections_types=["PEDESTRIAN", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE", "GENERIC_OBJECT"])
        else:
            raise ValueError(f"Invalid experiment type: {experiment}")
            
        # Simulation Manager
        # NOTE 控制仿真时间步长，确保仿真以固定时间间隔推进。
        simulation_time_controller = StepSimulationTimeController(scenario)

        # Stateful callbacks
        # NOTE MetricCallback: 记录仿真过程中的评估指标。
        # NOTE SimulationLogCallback: 保存仿真日志（格式为 msgpack）。
        metric_callback = MetricCallback(metric_engine=metric_engine)
        sim_log_callback = SimulationLogCallback(output_dir, simulation_dir, "msgpack")

        # Construct simulation and manager
        # NOTE 配置仿真环境，包括时间控制器、感知模块、控制器和场景。
        simulation_setup = SimulationSetup(
            time_controller=simulation_time_controller,
            observations=observations,
            ego_controller=ego_controller,
            scenario=scenario,
        )
        # NOTE 创建仿真对象，并绑定回调函数
        simulation = Simulation(
            simulation_setup=simulation_setup,
            callback=MultiCallback([metric_callback, sim_log_callback])
        )

        # Begin simulation
        # 运行仿真。
        simulation_runner = SimulationRunner(simulation, planner)
        # 执行仿真，返回运行报告（report）
        report = simulation_runner.run()
        runner_reports.append(report)
    
    # save reports
    save_runner_reports(runner_reports, output_dir, 'runner_reports')

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0

    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            # 打印失败仿真的错误信息。
            print("Failed Simulation.\n '%s'", result.error_message)
            failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    number_of_failures = len(scenarios) - number_of_successful
    print(f"Number of successful simulations: {number_of_successful}")
    print(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        # NOTE 收集失败场景的标识符。
        print(f"Failed simulations [log, token]:\n{failed_simulations}")
    
    print('Finished running simulations!')
    # NOTE 收集仿真报告（runner_reports），统计成功和失败的场景数量，并打印失败场景的详细信息。
    return runner_reports


def build_nuboard(scenario_builder, simulation_path):
    nuboard = NuBoard(
        nuboard_paths=simulation_path,
        scenario_builder=scenario_builder,
        vehicle_parameters=get_pacifica_parameters(),
        port_number=5106
    )

    nuboard.run()


def main(args):
    # parameters
    experiment_name = args.test_type  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    pth_name = os.path.basename(args.model_path)
    logging.info(f"pth is {pth_name}")

    job_name = 'DTPP_planner' + "_" + pth_name
    experiment_time = datetime.datetime.now()
    experiment = f"{experiment_name}/{job_name}/{experiment_time}"  
    # NOTE 和时间有关，每次实验的结果不会相互覆盖
    output_dir = f"testing_log/{experiment}"
    simulation_dir = "simulation"
    metric_dir = "metrics"
    # 聚合指标的子目录
    aggregator_metric_dir = "aggregator_metric"

    # initialize planner
    # 关闭梯度计算：因为规划器只需要推理，不需要训练
    torch.set_grad_enabled(False)
    # Planner 是一个自定义类（可能是用户实现的），根据提供的模型路径和设备加载规划器。
    # 自定义的规划器类，负责加载规划器模型。通过 args.model_path 指定模型路径，通过 args.device 指定运行设备（如 cuda 或 cpu）。
    planner = Planner(model_path=args.model_path, device=args.device)
    # NOTE 初始化评估回调------------------------------------------------------------------------------
    # initialize main aggregator 指标聚合
    metric_aggregators = build_metrics_aggregators(experiment_name, output_dir, aggregator_metric_dir)
    metric_save_path = f"{output_dir}/{metric_dir}"
    # NOTE 负责聚合多个场景的评估指标。
    metric_aggregator_callback = MetricAggregatorCallback(metric_save_path, metric_aggregators)
    # 将评估指标保存为文件
    metric_file_callback = MetricFileCallback(metric_file_output_path=f"{output_dir}/{metric_dir}",
                                              scenario_metric_paths=[f"{output_dir}/{metric_dir}"],
                                              delete_scenario_metric_files=True)
    # 生成评估指标的摘要和可视化报告。
    metric_summary_callback = MetricSummaryCallback(metric_save_path=f"{output_dir}/{metric_dir}",
                                                    metric_aggregator_save_path=f"{output_dir}/{aggregator_metric_dir}",
                                                    summary_output_path=f"{output_dir}/summary",
                                                    num_bins=20, pdf_file_name='summary.pdf')
    main_callbacks = MultiMainCallback([metric_file_callback, metric_aggregator_callback, metric_summary_callback])
    # 在仿真开始前调用，初始化评估回调。
    main_callbacks.on_run_simulation_start()

    # build simulation folder
    build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir)

    # build scenarios
    print('Extracting scenarios...')
    map_version = "nuplan-maps-v1.0"
    # 定义场景与地图的映射关系。
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    # NOTE 用于加载仿真场景，指定数据路径（args.data_path）和地图路径（args.map_path）。。
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version, scenario_mapping=scenario_mapping)
    # 场景过滤器：
    #     如果指定了 --load_test_set，则从 test_scenario.yaml 文件加载过滤参数。
    #     否则，使用默认的过滤参数（get_filter_parameters）。
    if args.load_test_set:
        params = yaml.safe_load(open('./test_scenario.yaml', 'r'))
        # NOTE **params是Python的解包操作符，它将字典params中的键值对作为关键字参数传递给ScenarioFilter类的构造函数。
        scenario_filter = ScenarioFilter(**params)
    else:
        # NOTE *用于解包一个序列（如列表或元组）。
        scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type))
    # 并行加载：使用 SingleMachineParallelExecutor 并行加载场景。
    worker = SingleMachineParallelExecutor(use_process_pool=False)
    scenarios = builder.get_scenarios(scenario_filter, worker)

    # begin testing
    print('Running simulations...')
    # NOTE 核心函数，负责运行仿真
    # 根据 experiment_name 设置不同的控制器和感知模块：
    #     LogPlaybackController：回放日志的控制器。
    #     TwoStageController：两阶段控制器，结合 LQR 追踪器和运动学模型。
    #     IDMAgents：基于 IDM（智能驾驶模型）的交互式感知模块。
    build_simulation(experiment_name, planner, scenarios, output_dir, simulation_dir, metric_dir)
    # NuBoard：一个可视化工具，用于展示仿真结果和评估指标。
    # 仿真结果加载：从实验输出文件夹中找到 .nuboard 文件，并将其传递给 NuBoard。
    # NOTE 在仿真结束后调用，处理评估回调。
    main_callbacks.on_run_simulation_end()
    # 从实验输出目录中找到 .nuboard 文件。
    simulation_file = [str(file) for file in pathlib.Path(output_dir).iterdir() if file.is_file() and file.suffix == '.nuboard']

    # show metrics and scenarios
    # 启动 NuBoard 可视化工具，展示仿真结果和评估指标
    build_nuboard(builder, simulation_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--map_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_type', type=str, default='closed_loop_nonreactive_agents')
    parser.add_argument('--load_test_set', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scenarios_per_type', type=int, default=20)
    args = parser.parse_args()

    main(args)
