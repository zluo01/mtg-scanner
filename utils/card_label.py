from dataclasses import dataclass


@dataclass
class CardLabel:
    name: str
    setCode: str
    number: str
    layout: str
    isOldSet: int

    def __post_init__(self):
        str_fields = ["name", "setCode", "number", "layout"]
        for field in str_fields:
            value = getattr(self, field)
            if not value or not value.strip():
                raise ValueError(f"{field} cannot be empty")

        if self.isOldSet is None:
            raise ValueError("isOldSet cannot be null")
