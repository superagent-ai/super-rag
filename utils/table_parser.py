from html.parser import HTMLParser


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_thead = False
        self.in_tbody = False
        self.in_row = False
        self.in_cell = False
        self.title_row = ""
        self.current_row = ""
        self.rows = []
        self.capture_next_row_as_title = True

    def handle_starttag(self, tag, _attrs):
        if tag == "table":
            self.in_table = True
        elif tag == "thead":
            self.in_thead = True
        elif tag == "tbody":
            self.in_tbody = True
        elif tag == "tr":
            self.in_row = True
            self.current_row = ""
        elif tag in ["td", "th"]:
            self.in_cell = True
            self.current_row += "<" + tag + ">"

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
        elif tag == "thead":
            self.in_thead = False
        elif tag == "tbody":
            self.in_tbody = False
        elif tag == "tr":
            self.in_row = False
            self.current_row += "</tr>"
            if self.capture_next_row_as_title:
                self.title_row = self.current_row
                self.capture_next_row_as_title = False
            else:
                self.rows.append(self.current_row)
        elif tag in ["td", "th"]:
            self.in_cell = False
            self.current_row += "</" + tag + ">"

    def handle_data(self, data):
        if self.in_cell:
            self.current_row += data
