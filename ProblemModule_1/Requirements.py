from datetime import datetime


class Requirement:
    """
    A class to represent a requirement.
    ...

    Attributes
    ----------
    id : str
        timestamp to uniquely identify each requirement
    text : str (default=None)
        plain text of the requirement, for human-readability
    symbol : str (default=None)
        characters to represent the constraint parameter in equations
    values : list (default=None)
        values which the requirement parameter is allowed to take and still
        meet customer and engineering needs

    Methods
    -------
    __get_text(text=None):
        gets requirement text from user and sets text attr
    __get_symbol(symbol=None):
        gets requirement symbol from user and sets symbol attr
    __get_values(values=None):
        gets requirement values from user and sets values attr
    """

    def __init__(self, text=None, symbol=None, values=None):
        self.id = datetime.now().strftime("%y%m%d%H%M%S")
        self.text = None
        self.symbol = None
        self.values = None
        self.__get_text(text=text)
        self.__get_symbol(symbol=symbol)
        self.__get_values(values=values)
        self.__prev = None
        self.__next = None

    def __repr__(self):
        return f"Requirement(id: {self.id}, symbol: {self.symbol})"

    def __str__(self):
        return self.text

    def __get_text(self, text=None):
        """
        Gets requirement text from user and sets text attr.
            params:
                text: plain text of the requirement, for human-readability
            return:
                None
        """
        if text:
            self.text = text
        else:
            self.text = input("Enter text for this constraint: ")

        assert isinstance(self.text, str), "text must be a string"

    def __get_symbol(self, symbol=None):
        """
        Gets symbol from user and sets symbol attr.
            params:
                symbol: chars to represent constraint parameter in equations
            return:
                None
        """
        if symbol:
            self.symbol = symbol
        else:
            self.symbol = input("Enter a symbol for the constraint "
                                "parameter: ")

        assert isinstance(self.symbol, str)

    def __get_values(self, values=None):
        """
        Gets requirement values from user and sets values attr.
            params:
                values: allowable values for the constraint parameter
            return:
                None
        """
        if values:
            self.values = values
        else:
            self.values = eval(input("Enter the allowable range or set: "))

        assert type(self.values) in [set, list], \
            "values must be a set or a list"

        assert all([type(el) in [float, int] for el in
                    list(self.values)]), \
            'members of values must be numeric'


class RequirementSet:
    def __init__(self):
        self.__head = None
        self.__cursor = None
        self.__size = 0

    def _add_requirement(self, text=None, symbol=None, values=None):
        newReq = Requirement(text=text, symbol=symbol, values=values)
        if not self.__head:
            self.__head = newReq
        else:
            cursor = self.__tail
            cursor._Requirement__next = newReq
            newReq._Requirement__prev = cursor
        self.__tail = newReq
        self.__size += 1

    def _batch_add(self, file):
        with open(file, "r") as f:
            lines = f.readlines()

        idx = list(range(len(lines)))[::4]

        for i in idx:
            text = lines[i].replace("\n", " ").strip()
            symbol = lines[i+1].replace("\n", " ").strip()
            values = eval(lines[i+2])
            self._add_requirement(text=text, symbol=symbol, values=values)

    def __getitem__(self, position):
        assert position < self.__size, "Index out of range"

        cursor = self.__head
        for _ in range(position):
            cursor = cursor._Requirement__next

        return cursor

    def __iter__(self):

        cursor = self.__head
        while cursor is not None:
            yield cursor
            cursor = cursor._Requirement__next

    def __len__(self):
        return self.__size

    def __call__(self):
        print("\nRequirements List:")

        cursor = self.__head
        flag = True
        while flag:
            flag = True if cursor._Requirement__next else False
            print("\t", cursor.symbol + ": ", cursor.text, cursor.values)
            cursor = cursor._Requirement__next
        print()

    def __repr__(self):
        return f"RequirementSet(len: {len(self)})"

    def __str__(self):
        string = ""

        for req in self:
            string += req.text + "\n"

        return string.strip()
