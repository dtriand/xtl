import math
import secrets
import uuid


_alphanumeric = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


class UUIDFactory:

    def __init__(self, alphabet: str = _alphanumeric,
                 sort_alphabet: bool = True):
        """
        Simple factory for creating UUIDs using a custom alphabet.

        :param alphabet: Custom alphabet to use for encoding and decoding UUIDs.
        :param sort_alphabet: If True, the alphabet will be sorted before use
            (default is True).
        """
        if not isinstance(alphabet, str):
            raise TypeError('\'alphabet\' must be a string')

        if sort_alphabet:
            self._alphabet = list(sorted(set(alphabet)))
        else:
            self._alphabet = [letter for letter in alphabet]

        self._alphabet_len = len(self._alphabet)

    @property
    def alphabet(self) -> str:
        """
        The alphabet used for encoding and decoding UUIDs.
        """
        return ''.join(self._alphabet)

    @property
    def length(self) -> int:
        """
        Length of the UUID, defaults to the number of digits required to
        represent a 128-bit UUID in the given alphabet.
        """
        return int(math.ceil(math.log(2**128, self._alphabet_len)))

    def encode(self, u: uuid.UUID) -> str:
        """
        Encode a UUID into a string using the custom alphabet.
        """
        if not isinstance(u, uuid.UUID):
            raise TypeError('\'u\' must be a UUID instance')

        output = ''
        num = u.int
        while num:
            num, rem = divmod(num, self._alphabet_len)
            output += self._alphabet[rem]
        if len(output) < self._length:
            output = output.ljust(self._length, self._alphabet[0])
        return output[::-1]

    def decode(self, s: str) -> uuid.UUID:
        """
        Decode a string back into a UUID using the custom alphabet.
        """
        if not isinstance(s, str):
            raise TypeError('\'s\' must be a string')

        num = 0
        for char in s:
            if char not in self._alphabet:
                raise ValueError(f'Character {char} not in alphabet')
            num = num * self._alphabet_len + self._alphabet.index(char)

        return uuid.UUID(int=num)

    def uuid(self, value: str = None) -> str:
        """
        Generate a UUID based on a given value or create a random one if
        no value is provided. If the value is a URL, it uses UUID5 with
        `uuid.NAMESPACE_URL`, otherwise it uses `uuid.NAMESPACE_DNS`.

        :param value: Optional string to seed the UUID generation.
        """
        if value is None:
            u = uuid.uuid4()
        elif value.lower().startswith(('http://', 'https://')):
            u = uuid.uuid5(uuid.NAMESPACE_URL, value)
        else:
            u = uuid.uuid5(uuid.NAMESPACE_DNS, value)
        return self.encode(u)

    def random(self, length: int = None) -> str:
        """
        Generate a random string of the specified length using the custom
        alphabet. If no length is specified, it defaults to the length of
        the UUID in the custom alphabet.
        """
        if length is None:
            length = self.length
        return ''.join(secrets.choice(self._alphabet) for _ in range(length))
