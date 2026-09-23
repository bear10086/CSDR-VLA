from common import register_framework
register_framework()
from deployment.model_server.server_policy import build_argparser, main
if __name__ == '__main__':
    main(build_argparser().parse_args())
