import functools
import datetime
import numpy as np
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from daxbench.core.engine.cloth_simulator import ClothState

# def log_method_call(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         print(f"[{timestamp}] Calling method: {func.__name__} of class: {args[0].__class__.__name__}")
#         return func(*args, **kwargs)

#     return wrapper


def log_method_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def format_value(value):
            # 値がndarrayの場合、そのshapeを出力
            if isinstance(value, np.ndarray) or isinstance(value, jnp.ndarray):
                return f"ndarray{value.shape}"
            # 値がClothStateの場合、その中身を出力
            if isinstance(value, ClothState):
                # 変数をイテレーション
                ret = "ClothState: "
                for k, v in value._asdict().items():
                    ret += f"{k}={format_value(v)}, "
                return ret

            # 値がNamedTupleの場合、その中身を出力
            if isinstance(value, NamedTuple):
                # 変数をイテレーション
                ret = "NamedTuple: "
                for k, v in value._asdict().items():
                    ret += f"{k}={format_value(v)}, "
                return ret

            # 値がTupleの場合、その中身を出力
            if isinstance(value, Tuple):
                # 変数をイテレーション
                ret = "Tuple: "
                for v in value:
                    ret += f"{format_value(v)}, "
                return ret

            # 値がDictの場合、その中身を出力
            if isinstance(value, dict):
                # 変数をイテレーション
                ret = "Dict: "
                for k, v in value.items():
                    ret += f"{k}={format_value(v)}, "
                return ret

            return repr(value)

        # 引数のログ出力
        arg_str = ", ".join(
            [format_value(arg) for arg in args[1:]] + [f"{k}={format_value(v)}" for k, v in kwargs.items()]
        )
        print(
            f"[{timestamp}] Calling method: {func.__name__} of class: {args[0].__class__.__name__} with arguments: ({arg_str})"
        )

        result = func(*args, **kwargs)

        # 戻り値のログ出力
        print(
            f"[{timestamp}] Method {func.__name__} of class {args[0].__class__.__name__} returned: {format_value(result)}"
        )

        return result

    return wrapper


def log_all_methods(cls):
    for name, method in vars(cls).items():
        if callable(method) and not name.startswith("__"):
            setattr(cls, name, log_method_call(method))
    return cls


@log_all_methods
class MyClass:
    def my_method_1(self, arg):
        print(f"my_method_1 called with {arg}")
        return "end"

    def my_method_2(self):
        print("my_method_2 called")


if __name__ == "__main__":
    my_class = MyClass()
    my_class.my_method_1("test")
    my_class.my_method_2()
