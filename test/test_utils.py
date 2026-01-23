import pytest

from thermal_solver.utils import NameGenerator


def test_name_generator():

    # Start with number
    with pytest.raises(ValueError):
        NameGenerator.register_name('1')

    # Empty
    with pytest.raises(ValueError):
        NameGenerator.register_name('')

    # Start with number
    with pytest.raises(ValueError):
        NameGenerator.new('1a')

    # Empty
    with pytest.raises(ValueError):
        NameGenerator.new('')

    # Register a valid name
    assert not NameGenerator.is_taken('hello')
    NameGenerator.register_name('hello')
    assert NameGenerator.is_taken('hello')

    # Register a name twice
    with pytest.raises(ValueError):
        NameGenerator.register_name('hello')

    # Create a new name
    a_new_name = NameGenerator.new(prefix='abc')
    assert a_new_name == 'abc000'
    assert NameGenerator.is_taken(a_new_name)

    # Create new name with the same prefix
    another_new_name = NameGenerator.new(prefix='abc')
    assert another_new_name == 'abc001'
    assert NameGenerator.is_taken(another_new_name)

    # Create new name with a suffix
    new_name_with_suffix = NameGenerator.new(prefix='abc', suffix='xyz')
    assert new_name_with_suffix == 'abc000xyz'
    assert NameGenerator.is_taken(new_name_with_suffix)

    another_new_name_with_suffix = NameGenerator.new(
        prefix='abc', suffix='xyz')
    assert another_new_name_with_suffix == 'abc001xyz'
    assert NameGenerator.is_taken(another_new_name_with_suffix)

    some_a = NameGenerator.register_or_create(None, 'a_')
    assert some_a == 'a_000'
    assert NameGenerator.is_taken(some_a)

    some_b = NameGenerator.register_or_create('b', 'a_')
    assert some_b == 'b'  # Name specified, no prefix or no suffix added.
    assert NameGenerator.is_taken(some_b)

    with pytest.raises(ValueError):
        # Registering existing name 'b'
        NameGenerator.register_or_create('b', 'a_')
