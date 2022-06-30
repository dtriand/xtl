from dataclasses import dataclass

@dataclass
class Attribute:

    name: str
    type: str
    description: str


class AttributeGroup:

    @property
    def children(self):
        return [str(c) for c in self.__dict__]
