# -*- coding: utf-8 -*-

import os.path

from fabopsy_lib import api


ROOT = os.path.dirname(os.path.abspath(__file__))


class OnnxModel(api.Model):
    def __init__(self, name: str):
        self.__name = name
        self.__path = os.path.join(ROOT, name)

    def exists(self) -> bool:
        return os.path.exists(self.__path)

    def cache(self) -> str:
        if self.exists():
            return self.__path

        raise RuntimeError(
            f'Unable to download on my own. '
            f'Need to contact the developer to obtain {self.__name} and place it in the directory {ROOT}')


def load() -> api.Model:
    return OnnxModel('heartrate.onnx')


def main():
    pass


if __name__ == '__main__':
    main()
