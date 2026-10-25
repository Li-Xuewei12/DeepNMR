TYPE_NAMES = ("1JHC", "1JHN", "2JHH", "2JHC", "2JHN", "3JHH", "3JHC", "3JHN")
TYPE_TO_ID = {name: index for index, name in enumerate(TYPE_NAMES)}
COMPONENT_NAMES = ("total", "FC", "SD", "PSO", "DSO")

CHIRAL_TO_ID = {
    "CHI_UNSPECIFIED": 0,
    "CHI_TETRAHEDRAL_CW": 1,
    "CHI_TETRAHEDRAL_CCW": 2,
    "CHI_OTHER": 3,
}
CHIRAL_UNKNOWN = 4

BOND_TO_ID = {
    "SINGLE": 0,
    "DOUBLE": 1,
    "TRIPLE": 2,
    "AROMATIC": 3,
    "QUADRUPLE": 4,
    "DATIVE": 5,
    "IONIC": 6,
}
BOND_UNKNOWN = 7


def charge_id(value: int) -> int:
    value = int(value)
    return value + 5 if -5 <= value <= 5 else 11

