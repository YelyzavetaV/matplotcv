import re
import math

coordinate_patterns = {
    'scientific': r'^[+-]?\d+(\.\d+)?([eE\^][+-]?\d+)?$',
}


def standard_coordinate(coordinate: str):
    coordinate = coordinate.strip()

    for notation, pattern in coordinate_patterns.items():
        if re.match(pattern, coordinate):
            break
    else:
        raise ValueError(
            f'Coordinate {coordinate} does not match any known notation.'
        )

    match notation:
        case 'scientific':
            coordinate = coordinate.replace('^', '**')
            if '**' in coordinate:
                coordinate = coordinate.split('**')
                coordinate = math.pow(
                    float(coordinate[0]), float(coordinate[1])
                )

            return coordinate
        case _:  # Shouldn't happen
            pass
