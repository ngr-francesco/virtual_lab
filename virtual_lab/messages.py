from virtual_lab.const import MsgTypes

class MessageDispatcher:
    def __init__(self, receiver):
        self.receiver = receiver

    def dispatch_message(self, msg, **kwargs):
        self.receiver.msg_handler.handle_msg(msg, **kwargs)


class MessageHandler:
    def __init__(self, owner, reactions: dict):
        self.owner = owner
        self.reactions = reactions
    
    def handle_msg(self, msg, **kwargs):
        try:
            reaction = getattr(self.owner, self.reactions[msg])
        except:
            # This is mainly for debugging, the user should never even think about messages.
            raise ValueError(f"The reaction of {self.owner.name} to msg {msg} is not defined.")
        return reaction(**kwargs)

        