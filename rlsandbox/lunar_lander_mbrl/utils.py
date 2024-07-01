import re


def assert_shape(array, shape) -> None:
    assert array.shape == shape, (array.shape, shape)


def every(n: int, i: int) -> bool:
    return (i + 1) % n == 0


def printne(*args, **kwargs) -> None:
    """print with no ending newline character."""
    print(*args, end='', flush=True, **kwargs)


def sanitize_name(
        name: str,
        valid_char_pattern: str = r'a-zA-Z0-9_\-',
        replacement_char: str = '-',
) -> str:
    name = re.sub(f'[^{valid_char_pattern}]', replacement_char, name)

    double_rc = replacement_char * 2
    while double_rc in name:
        name = name.replace(double_rc, replacement_char)

    if name.startswith(replacement_char):
        name = name[1:]
    if name.endswith(replacement_char):
        name = name[:-1]

    return name


if __name__ == '__main__':
    print(sanitize_name('this is a (test) name!'))
