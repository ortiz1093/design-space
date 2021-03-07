from datetime import datetime


class Requirement:
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

    def __get_text(self, text=None):
        if text:
            self.text = text
        else:
            self.text = input("Enter text for this constraint: ")

        assert isinstance(self.text, str), "text must be a string"

    def __get_symbol(self, symbol=None):
        if symbol:
            self.symbol = symbol
        else:
            self.symbol = input("Enter a symbol for the constraint "
                                "parameter: ")

        assert isinstance(self.symbol, str)

    def __get_values(self, values=None):
        if values:
            self.values = values
        else:
            self.values = eval(input("Enter the allowable range or set: "))

        assert type(self.values) in [set, list], \
            "values must be a set or a list"

        assert all([type(el) in [float, int] for el in
                    list(self.values)]), \
            'members of values must be numeric'


class DesignVariable:
    def __init__(self, description=None, symbol=None, space_type=None,
                 search_space=None):

        self.description = None
        self.symbol = None
        self.space_type = None
        self.search_space = None

        def __get_description(self, description=None):
            if description:
                assert isinstance(description, str), \
                    "variable description must be string"
                self.description = description

        assert isinstance(symbol, str), \
            "variable symbol must be string"

        assert space_type in ['continuous', 'discrete', 'explicit'], \
            "space_type must be either 'continuous', 'discrete', or 'explicit'"

        assert type(search_space) in ['list', 'set'], \
            "search_space must be of type 'list' or 'set'"

        self.description = description
        self.symbol = symbol
        self.space_type = space_type
        self.search_space = search_space


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

    def __len__(self):
        return self.__size

    def __call__(self):
        print("\nRequirements List:")

        cursor = self.__head
        flag = True
        while flag:
            flag = True if cursor._Requirement__next else False
            print("\t", cursor.text)
            cursor = cursor._Requirement__next
        print()


class Problem:
    def __init__(self):
        self.A = []
        self.B = []
        self.M = []
        self.P = None
        self.S = None
        self.Omega = None
        self.R = RequirementSet()

    def add_requirement(self, text=None, symbol=None, values=None, file=None):
        if file:
            assert all([not text, not symbol, not values]), \
                "Cannot provide file and individual requirement"
            self.R._batch_add(file)
            self.__generate_param_set()
        else:
            self.R._add_requirement(text=text, symbol=symbol,
                                    values=values)

        self.__generate_param_set()

    def get_requirements_list(self):
        return [req for req in self.R]

    def __generate_param_set(self):

        cursor = self.R._RequirementSet__head
        flag = True
        while flag:
            flag = True if cursor._Requirement__next else False
            if cursor.symbol not in self.A:
                self.A.append(cursor.symbol)
            else:
                pass
            cursor = cursor._Requirement__next

        assert len(self.R) == len(set(self.A)), "Number of requirements " \
            "doesn't match number of symbols. Possible duplicate."

    def show_param_set(self):
        print("A = ", self.A)


if __name__ == "__main__":
    printer = Problem()
    printer.add_requirement(file="reqs.txt")

    text3 = "This is requirement 3"
    symbol3 = "a3"
    values3 = [1, 2, 3, 4]
    printer.add_requirement(text=text3, symbol=symbol3, values=values3)

    text4 = "This is requirement 4"
    symbol4 = "a4"
    values4 = [1, 4]
    printer.add_requirement(text=text4, symbol=symbol4, values=values4)

    printer.R()
    printer.show_param_set()
    print("Number of requirements:", len(printer.R))
