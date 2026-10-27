
import re


def convert_to_markdown(text):
    
    if not text or not text.strip():
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]

    formatted_lines = []
    previous_was_blank = True

    for line in lines:
        if not line:
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            previous_was_blank = True
            continue

        if re.match(r"^#{1,6}\s+", line):
            formatted_lines.append(line)
            previous_was_blank = False
            continue

        if re.match(r"^[-*+]\s+", line):
            formatted_lines.append(line)
            previous_was_blank = False
            continue

        if re.match(r"^\d+[.)]\s+", line):
            formatted_lines.append(line)
            previous_was_blank = False
            continue

        letters = [char for char in line if char.isalpha()]
        is_mostly_uppercase = (
            len(letters) >= 3
            and sum(char.isupper() for char in letters) / len(letters) >= 0.85
        )

        if len(line) <= 70 and is_mostly_uppercase:
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")

            formatted_lines.append(f"## {line.title()}")
            formatted_lines.append("")
            previous_was_blank = True
            continue

        numbered_match = re.match(r"^(\d+)[.)]\s*(.+)$", line)

        if numbered_match:
            number, item_text = numbered_match.groups()
            formatted_lines.append(f"{number}. {item_text}")
            previous_was_blank = False
            continue

        bullet_match = re.match(r"^[•●▪◦]\s*(.+)$", line)

        if bullet_match:
            formatted_lines.append(f"- {bullet_match.group(1)}")
            previous_was_blank = False
            continue

        if re.match(r"^[\w /()-]{1,40}:\s+\S", line):
            formatted_lines.append(line)
            previous_was_blank = False
            continue

        formatted_lines.append(line)
        previous_was_blank = False

    while formatted_lines and formatted_lines[-1] == "":
        formatted_lines.pop()

    return "\n".join(formatted_lines) + "\n"
