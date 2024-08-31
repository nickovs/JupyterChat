import asyncio
import uuid
from typing import List

import ipywidgets as widgets
import markdown2

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import Runnable

_LABEL_COLOURS = {
    "user": "green",
    "system": "red",
    "agent": "blue",
    "tool": "orange"
}

_CODE_STYLE_INFO = {
    '': {'bg': '#f8f8f8'},
    'bp': {'c': '#008000'},
    'c': {'c': '#408080', 's': 'italic'},
    'c1': {'c': '#408080', 's': 'italic'},
    'ch': {'c': '#408080', 's': 'italic'},
    'cm': {'c': '#408080', 's': 'italic'},
    'cp': {'c': '#BC7A00'},
    'cpf': {'c': '#408080', 's': 'italic'},
    'cs': {'c': '#408080', 's': 'italic'},
    'dl': {'c': '#BA2121'},
    'err': {'br': '1px solid #FF0000'},
    'fm': {'c': '#0000FF'},
    'gd': {'c': '#A00000'},
    'ge': {'s': 'italic'},
    'gh': {'c': '#000080', 'w': 'bold'},
    'gi': {'c': '#00A000'},
    'go': {'c': '#888888'},
    'gp': {'c': '#000080', 'w': 'bold'},
    'gr': {'c': '#FF0000'},
    'gs': {'w': 'bold'},
    'gt': {'c': '#0044DD'},
    'gu': {'c': '#800080', 'w': 'bold'},
    'hll': {'bg': '#ffffcc'},
    'il': {'c': '#666666'},
    'k': {'c': '#008000', 'w': 'bold'},
    'kc': {'c': '#008000', 'w': 'bold'},
    'kd': {'c': '#008000', 'w': 'bold'},
    'kn': {'c': '#008000', 'w': 'bold'},
    'kp': {'c': '#008000'},
    'kr': {'c': '#008000', 'w': 'bold'},
    'kt': {'c': '#B00040'},
    'm': {'c': '#666666'},
    'mb': {'c': '#666666'},
    'mf': {'c': '#666666'},
    'mh': {'c': '#666666'},
    'mi': {'c': '#666666'},
    'mo': {'c': '#666666'},
    'na': {'c': '#7D9029'},
    'nb': {'c': '#008000'},
    'nc': {'c': '#0000FF', 'w': 'bold'},
    'nd': {'c': '#AA22FF'},
    'ne': {'c': '#D2413A', 'w': 'bold'},
    'nf': {'c': '#0000FF'},
    'ni': {'c': '#999999', 'w': 'bold'},
    'nl': {'c': '#A0A000'},
    'nn': {'c': '#0000FF', 'w': 'bold'},
    'no': {'c': '#880000'},
    'nt': {'c': '#008000', 'w': 'bold'},
    'nv': {'c': '#19177C'},
    'o': {'c': '#666666'},
    'ow': {'c': '#AA22FF', 'w': 'bold'},
    's': {'c': '#BA2121'},
    's1': {'c': '#BA2121'},
    's2': {'c': '#BA2121'},
    'sa': {'c': '#BA2121'},
    'sb': {'c': '#BA2121'},
    'sc': {'c': '#BA2121'},
    'sd': {'c': '#BA2121', 's': 'italic'},
    'se': {'c': '#BB6622', 'w': 'bold'},
    'sh': {'c': '#BA2121'},
    'si': {'c': '#BB6688', 'w': 'bold'},
    'sr': {'c': '#BB6688'},
    'ss': {'c': '#19177C'},
    'sx': {'c': '#008000'},
    'vc': {'c': '#19177C'},
    'vg': {'c': '#19177C'},
    'vi': {'c': '#19177C'},
    'vm': {'c': '#19177C'},
    'w': {'c': '#bbbbbb'}
}


_CODE_CSS = None


def _get_code_css():
    global _CODE_CSS
    if _CODE_CSS is None:
        style_items = {
            'bg': 'background',
            'br': 'border',
            'c': 'color',
            's': 'font-style',
            'w': 'font-weight'
        }
        unpacked = "\n".join(
            ".codehilite" + (" ." + name if name else "") + " { " + "; ".join(
                style_items[k] + ": " + v
                for k, v in style_info.items()
            ) + " }"
            for name, style_info in _CODE_STYLE_INFO.items()
        )
        _CODE_CSS = f"<style>\n{unpacked}\n</style>\n"
    return _CODE_CSS


class ThreadItem:
    def __init__(self, text: str, sender: str, tool_name=None):
        if sender not in _LABEL_COLOURS:
            raise ValueError(f"Unknown sender: {sender}")
        self._sender = sender
        self._text = text
        self._tool_name = tool_name
        self._widget = widgets.HTML(self.html, description=self.description)

    @property
    def html(self):
        text = self._text
        if self._sender == "tool":
            name = self._tool_name if self._tool_name else "???"
            text = f"### Tool use for: `{name}`\n```json\n{text}\n```\n"
        else:
            text += "\n```\n" if text.count("\n```") % 2 else ""

        html = markdown2.markdown(text, extras=["fenced-code-blocks", "tables"])

        return html

    @property
    def description(self):
        return r"\(\color{" + _LABEL_COLOURS[self._sender] + "}{" + self._sender + r":}\)"

    @property
    def widget(self):
        return self._widget

    @property
    def sender(self):
        return self._sender

    @sender.setter
    def sender(self, sender: str):
        if sender not in _LABEL_COLOURS:
            raise ValueError(f"Unknown sender: {sender}")
        self._sender = sender
        self._widget.description = self.description

    @property
    def message(self):
        match self._sender:
            case "system":
                return SystemMessage(content=self._text)
            case "user":
                return HumanMessage(content=self._text)
            case "agent":
                return AIMessage(content=self._text)
            case "tool":
                return ToolMessage(content=self._text)

    def update(self, text, tool_name=None):
        self._text = text
        if tool_name:
            self._tool_name = tool_name
        self._widget.value = self.html

    def append_text(self, text, tool_name=None):
        self._text += text
        if tool_name:
            self._tool_name = tool_name
        self._widget.value = self.html


class ChatDisplay(widgets.VBox):
    def __init__(self, app: Runnable, debug: bool = False):
        self.app = app
        self._thread_id = uuid.uuid4()
        self.thread_items: List[ThreadItem] = []
        # The status label also serves to ensure we have the code CSS style, so it's HTML
        self.status_label = widgets.HTML("")
        self._status = ""
        self.thread = widgets.VBox(children=[])
        if debug:
            self.debug = widgets.Textarea(description="Debug")
        else:
            self.debug = None
        self.input = widgets.Textarea(placeholder="Enter your message:")
        self.button = widgets.Button(description="Send")
        self.button.on_click(self.send_message)
        self._message_task = None

        self.status = "Idle"

        children = [self.status_label, self.thread, self.input, self.button]
        if debug:
            children.append(self.debug)

        super().__init__(children=children)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
        self.status_label.value = _get_code_css() + "Status: " + value
        if self.debug:
            self.debug.value += value + "\n"

    def send_message(self, button):
        _ = button
        self.status = "User clicked"
        message = self.input.value
        self.status = f"Got message: {message}"
        self.input.value = ""
        item = self._push_new_message(text=message)

        self.status = "Message appended"

        run_loop = asyncio.get_event_loop()

        self.status = f"Have run loop: {run_loop}"

        self._message_task = run_loop.create_task(self.process_message(item))

        self.status = f"Created task: {self._message_task}"

    def _push_new_message(self, text="", sender="user", tool_name=None):
        item = ThreadItem(text=text, sender=sender, tool_name=tool_name)
        self.thread_items.append(item)
        self.thread.children = [item.widget for item in self.thread_items]
        return item

    async def process_message(self, message: ThreadItem):
        self.status = "Sending message"

        stream = self.app.astream_events(
            {"messages": [message.message]},
            config={"configurable": {"thread_id": self._thread_id}},
            version="v2"
        )

        self.status = f"Stream created: {stream}"

        current_message = None
        self.status = f"Stream handler entered: {stream}"

        async for event in stream:
            self.status = event["event"]
            if event['event'] == "on_chat_model_start":
                current_message = None
            elif event['event'] == "on_chat_model_stream":
                content = event['data']['chunk'].content
                for c in content:
                    if 'type' in c and c['type'] == "text":
                        if current_message is None or current_message.sender != "agent":
                            current_message = self._push_new_message(sender="agent")

                        current_message.append_text(c['text'])
                    elif 'type' in c and c['type'] == "tool_use":
                        if current_message is None or current_message.sender != "tool":
                            current_message = self._push_new_message(sender="tool", tool_name=c.get('name'))

                        if 'input' in c:
                            current_message.append_text(c['input'], tool_name=c.get('name'))
            elif event['event'] == "on_chat_model_end":
                # display(JSON(event))
                content = event['data']['output'].content
                c = content[-1]
                if 'type' in c and c['type'] == "text":
                    chat_output = c['text']
                    if current_message is None or current_message.sender != "agent":
                        current_message = self._push_new_message(sender="agent")
                    current_message.update(chat_output)
                elif 'type' in c and c['type'] == "tool_use":
                    if 'input' in c:
                        tool_output = c['input']
                        if current_message is None or current_message.sender != "tool":
                            current_message = self._push_new_message(sender="tool", tool_name=c.get('name'))
                        current_message.update(tool_output, tool_name=c.get('name'))
