from glayout.syntaxer2.Command import Command

class help(Command):
    name = "help"
    summary = "gives general/specific help"
    help = """help - gives general help about StrictSyntax\nhelp {command} - gives help about {command}"""

    def action(self):
        """prints out the supported commands, and summaries for them
        """
        words = self.line.split(" ")
        # if the command is just help, then it just wants general help
        gen_help = "StrictSyntax currently has support for these commands:\n"

        if len(words) == 1:
            for command in self.session.commands:
                gen_help += f"{command.name} - {command.summary}\n"

        gen_help += "To view specific help for a command, type \"help {command name}\""
        print(gen_help)

    def manipulate_code():
        """does nothing to manipulate the code, this is a command language specific command
        """
        pass


    


