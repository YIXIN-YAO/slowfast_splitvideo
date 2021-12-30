import os
import time
import grpc
from concurrent import futures

import pd_service.proto.split_pb2 as pb2
import pd_service.proto.split_pb2_grpc as pb2_grpc

from pd_service.lib.logger_mgr import LoggerMgr
from pd_service.alg_split.slowfast.config.defaults import assert_and_infer_cfg
from pd_service.alg_split.slowfast.utils.parser import load_config, parse_args
from pd_service.alg_split.demo_net import demo
from pd_service.alg_split.split import split_video
from pd_service.alg_split.yolo_det.detect import start_det
from pd_service.lib.srv_config import SrvConfig


class SplitVideoSrv(pb2_grpc.SplitVideoServicer):
    def __init__(self, logger=None):
        super(SplitVideoSrv, self).__init__()

        self.logger = logger if logger else LoggerMgr()

    def video_split(self, request, context):
        """
        GRPC
        :param request:
        :param context:
        :return:
        """
        video_path = request.video_path
        if os.path.exists(video_path):
            temp_video_path = start_det(video_path, 50, 50)  # 和其它地方用到的start_det函数略有不同;注意整合路径

            args = parse_args()
            args.cfg_file = "pd_service/alg_split/configs/SLOWFAST_4x16_R50.yaml"  # 配置文件路径
            cfg = load_config(args)
            cfg = assert_and_infer_cfg(cfg)
            cfg.DEMO.INPUT_VIDEO = temp_video_path  # 覆盖来自配置文件的值

            csv_path = demo(cfg)
            print("csv_path:", csv_path)
            out_dir = split_video(cfg, csv_path, video_path)  # 这个路径是YAML文件里的OUTPUT_FILE_DIR和视频名拼的
            os.remove(temp_video_path)  # 删掉剪切的视频

            return pb2.result_str(new_video_dir=out_dir,
                                  new_video_count=str(len([x for x in os.listdir(out_dir) if x.endswith('avi')])),
                                  status_code='200')
        else:
            return pb2.result_str(new_video_dir="", new_video_count='0', status_code="400")


def server():
    logger = LoggerMgr(file_log_level=int(SrvConfig.LOGGER_LEVEL), stream_log_level=int(SrvConfig.LOGGER_LEVEL))
    server_host = SrvConfig.VIDEO_SPLIT_SERVER_HOST
    server_port = SrvConfig.VIDEO_SPLIT_SERVER_PORT
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=48))   # todo
    pb2_grpc.add_SplitVideoServicer_to_server(SplitVideoSrv(logger), grpc_server)
    grpc_server.add_insecure_port(server_host + ':' + server_port)
    grpc_server.start()
    logger.info(f"Start video split service success, listen on {server_host}:{server_port}")
    try:
        while 1:
            time.sleep(86400)    # todo
    except KeyboardInterrupt:
        grpc_server.stop(0)


if __name__ == '__main__':
    server()





