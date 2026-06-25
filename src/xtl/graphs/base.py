from xtl.common.options import Options
from xtl.math.uuid import UUIDFactory


uuid: UUIDFactory = UUIDFactory()
random_uid = lambda: uuid.random(length=5)


class BaseGraphModel(Options):
    model_config = Options.model_config | {'use_enum_values': False}
