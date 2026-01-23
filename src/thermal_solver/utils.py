import numpy as np

from .vectors import versor


def _get_func_name_():
    import inspect
    return inspect.currentframe().f_back.f_code.co_name


def calculate_effective_area_factor(orientation_a, orientation_b) -> float:
    """Return the dot procut between the two versors of the orientation of the
    surfaces, inverted in sign (opposing for positive factor), and at least 0"""
    v1, v2 = versor(orientation_a), versor(orientation_b)
    return max(0, -np.dot(v1, v2))


class NameGenerator:

    _names_taken: list[str] = []
    _num_fmt: str = '03d'

    @classmethod
    def new(cls, prefix: str, suffix: str = '') -> str:
        i = 0
        name = prefix + format(i, cls._num_fmt) + suffix
        while cls.is_taken(name):
            i += 1
            name = prefix + format(i, cls._num_fmt) + suffix
        cls.register_name(name)
        return name

    @classmethod
    def is_taken(cls, name: str) -> bool:
        return name in cls._names_taken

    @classmethod
    def register_name(cls, name: str):
        if not name:
            raise ValueError(f'Name cannot be empty.')
        if not name[0].isalpha():
            raise ValueError(
                f'The first character of the name should be a letter. Not "{name[0]}"')
        if cls.is_taken(name):
            raise ValueError(f'Name already registered: {name}')
        cls._names_taken.append(name)

    @classmethod
    def register_or_create(cls, name: str | None, prefix: str, suffix: str = '') -> str:
        """Register or Create a new name depending on the argument 'name'.

        If name is not None and not empty, register name as it is.

        If name is None or empty, create a new name with the prefix/suffix specified.

        Returns:
            The registered name if name was registered. The created name if a new name was created.

        """
        if not name:
            return cls.new(prefix, suffix)
        cls.register_name(name)
        return name

    @classmethod
    def _clear(cls):
        # NOTE: only for testing purposes. It's not intended to be used!
        cls._names_taken.clear()
