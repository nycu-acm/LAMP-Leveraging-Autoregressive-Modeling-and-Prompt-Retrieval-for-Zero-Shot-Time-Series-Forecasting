from models import LAMP


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'LAMP': LAMP,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
