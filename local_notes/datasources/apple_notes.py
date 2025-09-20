import json
import subprocess
from typing import List

from ..models import Document
from ..utils.html import html_to_text


FIELD_SEP = "|||FIELD|||"
REC_SEP = "|||REC|||"


APPLE_SCRIPT_ALL = f'''
set _field to "{FIELD_SEP}"
set _rec to "{REC_SEP}"
set AppleScript's text item delimiters to _rec
set outRecords to {{}}

tell application "Notes"
    repeat with acc in accounts
        repeat with f in folders of acc
            repeat with n in notes of f
                set note_id to id of n
                set note_name to name of n
                set note_body to body of n
                set mod_date to modification date of n
                set folder_name to name of f
                set AppleScript's text item delimiters to _field
                set rec to note_id & _field & note_name & _field & folder_name & _field & (mod_date as string) & _field & note_body
                set end of outRecords to rec
            end repeat
        end repeat
    end repeat
end tell

set AppleScript's text item delimiters to _rec
return outRecords as text
'''


class AppleNotesDataSource:
    """Fetch notes from Apple Notes via AppleScript.

    Produces `Document` with text extracted from the HTML body.
    """

    def fetch(self) -> List[Document]:
        try:
            result = subprocess.run(
                ["osascript", "-e", APPLE_SCRIPT_ALL],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"AppleScript failed: {e.stderr.strip()}" )

        output = result.stdout
        if not output.strip():
            return []
        records = output.split(REC_SEP)
        docs: List[Document] = []
        for rec in records:
            if not rec:
                continue
            parts = rec.split(FIELD_SEP)
            if len(parts) < 5:
                continue
            note_id, name, folder, mod_date, body_html = parts[0], parts[1], parts[2], parts[3], FIELD_SEP.join(parts[4:])
            text = html_to_text(body_html)
            title = name or (text.split("\n", 1)[0] if text else "Untitled")
            metadata = {
                "folder": folder,
                "modified": mod_date,
            }
            docs.append(Document(id=note_id, title=title, text=text, source="apple-notes", metadata=metadata))
        return docs


# Additional AppleScripts for incremental flows
APPLE_SCRIPT_META = f'''
set _field to "{FIELD_SEP}"
set _rec to "{REC_SEP}"
set AppleScript's text item delimiters to _rec
set outRecords to {{}}

tell application "Notes"
    repeat with acc in accounts
        repeat with f in folders of acc
            repeat with n in notes of f
                set note_id to id of n
                set note_name to name of n
                set mod_date to modification date of n
                set folder_name to name of f
                set AppleScript's text item delimiters to _field
                set rec to note_id & _field & note_name & _field & folder_name & _field & (mod_date as string)
                set end of outRecords to rec
            end repeat
        end repeat
    end repeat
end tell

set AppleScript's text item delimiters to _rec
return outRecords as text
'''


APPLE_SCRIPT_BODIES = f'''
on run argv
    set _field to "{FIELD_SEP}"
    set _rec to "{REC_SEP}"
    set ids_csv to item 1 of argv
    set AppleScript's text item delimiters to ","
    set id_list to text items of ids_csv
    set AppleScript's text item delimiters to _rec
    set outRecords to {{}}

    tell application "Notes"
        repeat with acc in accounts
            repeat with f in folders of acc
                repeat with n in notes of f
                    set note_id to id of n as text
                    if id_list contains note_id then
                        set note_body to body of n
                        set AppleScript's text item delimiters to _field
                        set rec to note_id & _field & note_body
                        set end of outRecords to rec
                    end if
                end repeat
            end repeat
        end repeat
    end tell

    set AppleScript's text item delimiters to _rec
    return outRecords as text
end run
'''


class AppleNotesIncremental:
    def list_metadata(self) -> List[dict]:
        try:
            result = subprocess.run(
                ["osascript", "-e", APPLE_SCRIPT_META],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"AppleScript (meta) failed: {e.stderr.strip()}")

        output = result.stdout
        if not output.strip():
            return []
        items = []
        for rec in output.split(REC_SEP):
            if not rec:
                continue
            parts = rec.split(FIELD_SEP)
            if len(parts) < 4:
                continue
            items.append({
                "id": parts[0],
                "title": parts[1],
                "folder": parts[2],
                "modified": parts[3],
            })
        return items

    def fetch_bodies(self, ids: List[str]) -> List[dict]:
        if not ids:
            return []
        ids_csv = ",".join(ids)
        try:
            result = subprocess.run(
                ["osascript", "-e", APPLE_SCRIPT_BODIES, ids_csv],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"AppleScript (bodies) failed: {e.stderr.strip()}")
        output = result.stdout
        if not output.strip():
            return []
        out = []
        for rec in output.split(REC_SEP):
            if not rec:
                continue
            parts = rec.split(FIELD_SEP)
            if len(parts) < 2:
                continue
            out.append({"id": parts[0], "body_html": FIELD_SEP.join(parts[1:])})
        return out
