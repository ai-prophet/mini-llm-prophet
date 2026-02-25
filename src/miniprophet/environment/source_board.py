"""Source board: stateful evidence tracker for the forecasting agent."""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_SENTIMENTS = {"very_positive", "positive", "neutral", "negative", "very_negative"}

REACTION_COMPACT = {
    "very_positive": "++",
    "positive": "+",
    "neutral": "~",
    "negative": "-",
    "very_negative": "--",
}


@dataclass
class Source:
    """A single search result returned by a search tool."""

    url: str
    title: str
    snippet: str
    date: str | None = None


@dataclass
class BoardEntry:
    """A source on the board, annotated with the model's analytical note."""

    id: int
    source_id: str | None
    source: Source
    note: str
    reaction: dict[str, str] = field(default_factory=dict)


class SourceBoard:
    """Ordered collection of annotated sources that the agent builds up over time."""

    def __init__(self) -> None:
        self._entries: list[BoardEntry] = []
        self._next_id: int = 1

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self,
        source: Source,
        note: str,
        reaction: dict[str, str] | None = None,
        source_id: str | None = None,
    ) -> BoardEntry:
        entry = BoardEntry(
            id=self._next_id,
            source_id=source_id,
            source=source,
            note=note,
            reaction=reaction or {},
        )
        self._entries.append(entry)
        self._next_id += 1
        return entry

    def edit_note(
        self,
        board_id: int,
        new_note: str,
        reaction: dict[str, str] | None = None,
    ) -> BoardEntry:
        for entry in self._entries:
            if entry.id == board_id:
                entry.note = new_note
                if reaction is not None:
                    entry.reaction = reaction
                return entry
        raise KeyError(f"No board entry with id {board_id}")

    def get(self, board_id: int) -> BoardEntry:
        for entry in self._entries:
            if entry.id == board_id:
                return entry
        raise KeyError(f"No board entry with id {board_id}")

    def render(self) -> str:
        """Render the board as an XML-structured string for the model."""
        if not self._entries:
            return "<source_board>\n(empty)\n</source_board>"
        lines = [f'<source_board count="{len(self._entries)}">']
        for entry in self._entries:
            date_attr = f' date="{entry.source.date}"' if entry.source.date else ""
            lines.append(
                f'<source board_id="{entry.id}" title="{entry.source.title}" '
                f'url="{entry.source.url}"{date_attr}>'
            )
            lines.append(f"Note: {entry.note}")
            if entry.reaction:
                parts = [
                    f"{outcome} [{REACTION_COMPACT.get(s, '?')}]"
                    for outcome, s in entry.reaction.items()
                ]
                lines.append(f"Reactions: {'  '.join(parts)}")
            lines.append("</source>")
        lines.append("</source_board>")
        return "\n".join(lines)

    def serialize(self) -> list[dict]:
        return [
            {
                "id": e.id,
                "source_id": e.source_id,
                "source": {
                    "url": e.source.url,
                    "title": e.source.title,
                    "snippet": e.source.snippet,
                    "date": e.source.date,
                },
                "note": e.note,
                "reaction": e.reaction,
            }
            for e in self._entries
        ]
