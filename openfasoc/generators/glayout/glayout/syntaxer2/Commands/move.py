from glayout.syntaxer2.Command import Command

class move(Command):
    name = "move"
    summary = "moves components"
    help = "move {component 1} {direction} {component 2}"

    def action(self):
        """doesn't have any actions on the front end, this just manipulate code
        """
        pass

    def manipulate_code(self):
        """adds to the movement table
        """

        words = self.line.split(" ")
        irrelevant_words = ["to", "of", "the"]

        words = [for word in words if word not in irrelevant_words]

        direction_dict = {
            "left": 0, "west": 0,
            "right": 1, "east": 1,
            "above": 2, "north": 2,
            "below": 3, "south": 3
        }

        component1 = words[1]
        direction = direction_dict[words[2]]
        component2 = direction_dict[words[3]]
        seperation = None

        # if there is a seperation argument
        if len(words) > 4:
            seperation = words[6:]

        seperation = "".join(seperation)
        seperation = seperation.split("=")[-1]

        self.session.code.add_movement_code(component1, component2, direction, seperation)
