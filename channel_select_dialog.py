import json
import os

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QScrollArea, QWidget, QMessageBox
)


# The exact roles each model needs assigned before it can extract features.
REQUIRED_ROLES = {
    "1_LightGBM-2EEG": ["EEG1", "EEG2", "EMG"],
    "2_LightGBM-1EEG": ["EEG", "EMG"],
}

OTHER_ROLE_OPTIONS = ["Ignore", "Display Only"]

SELECT_PLACEHOLDER = "-- Select a channel --"

# Where "learned" channel-name -> role associations are remembered, so that
# once you've confirmed e.g. "1_EEG" -> EMG for one file, that exact
# assignment is offered again (never assumed) for future files that have a
# channel literally named "1_EEG".
_LEARNED_DEFAULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "channel_role_defaults.json"
)


def _load_learned_defaults():
    if os.path.exists(_LEARNED_DEFAULTS_PATH):
        try:
            with open(_LEARNED_DEFAULTS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_learned_defaults(defaults):
    with open(_LEARNED_DEFAULTS_PATH, "w") as f:
        json.dump(defaults, f, indent=2)


class ChannelSelectDialog(QDialog):
    """
    Shown once per EDF file (and cached to disk afterward).

    Structure is role-centric, NOT label-guessing: for each role the model
    needs (e.g. EEG1, EEG2, EMG), you get a dropdown listing every real
    channel name found in the EDF file, and you explicitly pick which one
    it is. Nothing is ever auto-assigned based on a channel's name containing
    the word "EEG" or "EMG" -- that was the previous design's flaw (two
    channels both literally named "EEG" both auto-guessed as "EEG", when one
    was actually EMG). The dropdown only pre-selects a channel if you've
    explicitly confirmed that exact channel name for that exact role before
    (see "learned defaults" below); otherwise it's left blank and you must
    choose.

    Any channel not used for a scoring role is separately classified as
    "Display Only" (shown in the viewer, not scored) or "Ignore" (not
    loaded).
    """

    def __init__(self, ch_names, model_name, filename="", parent=None):
        super().__init__(parent)
        self.ch_names = ch_names
        self.model_name = model_name
        self.required_roles = REQUIRED_ROLES.get(model_name, [])
        self.role_combo_boxes = {}   # role -> QComboBox of channel names
        self.other_combo_boxes = {}  # channel name -> QComboBox of Ignore/Display Only
        self.learned_defaults = _load_learned_defaults()

        self.setWindowTitle(f"Select Channels - {filename}")
        self.setMinimumWidth(500)
        self.setMinimumHeight(480)

        outer_layout = QVBoxLayout(self)

        header = QLabel(
            f"Model '{model_name}' needs: {', '.join(self.required_roles)}.\n\n"
            "For each role below, pick which actual channel from this file it is. "
            "Nothing is guessed from a channel's name -- you always choose explicitly, "
            "unless you've confirmed that exact channel name for that role before, in "
            "which case it's offered again as a starting point (you can still change it).\n\n"
            "Any channel not used above can be marked 'Display Only' (shown in the "
            "viewer, not scored) or 'Ignore' (not loaded)."
        )
        header.setWordWrap(True)
        outer_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        form_layout = QVBoxLayout(scroll_content)

        # --- Section 1: one dropdown per required scoring role ---
        section1_label = QLabel("Scoring channels (required):")
        section1_label.setStyleSheet("font-weight: bold;")
        form_layout.addWidget(section1_label)

        for role in self.required_roles:
            row = QHBoxLayout()
            label = QLabel(f"{role}:")
            label.setMinimumWidth(220)
            combo = QComboBox()
            combo.addItem(SELECT_PLACEHOLDER)
            combo.addItems(ch_names)

            default_ch = self._learned_channel_for_role(role)
            if default_ch in ch_names:
                combo.setCurrentText(default_ch)

            row.addWidget(label)
            row.addWidget(combo)
            form_layout.addLayout(row)
            self.role_combo_boxes[role] = combo

        # --- Section 2: every other channel, Ignore vs Display Only ---
        section2_label = QLabel("\nAll other channels:")
        section2_label.setStyleSheet("font-weight: bold;")
        form_layout.addWidget(section2_label)

        for ch in ch_names:
            row = QHBoxLayout()
            label = QLabel(ch)
            label.setMinimumWidth(220)
            combo = QComboBox()
            combo.addItems(OTHER_ROLE_OPTIONS)

            learned = self.learned_defaults.get(ch)
            if learned in OTHER_ROLE_OPTIONS:
                combo.setCurrentText(learned)
            elif any(tag in ch.lower() for tag in ["ttl", "laser", "opto", "stim", "trig"]):
                combo.setCurrentText("Display Only")

            row.addWidget(label)
            row.addWidget(combo)
            form_layout.addLayout(row)
            self.other_combo_boxes[ch] = combo

        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll)

        self.button_ok = QPushButton("Confirm Channel Selection")
        self.button_ok.clicked.connect(self.validate_and_accept)
        outer_layout.addWidget(self.button_ok)

    def _learned_channel_for_role(self, role):
        """Look up whether any channel in this file was previously
        confirmed for this exact role, e.g. learned_defaults["1_EEG"] ==
        "EMG" would make 1_EEG the default suggestion for the EMG role."""
        for ch in self.ch_names:
            if self.learned_defaults.get(ch) == role:
                return ch
        return None

    def validate_and_accept(self):
        role_selections = {role: combo.currentText() for role, combo in self.role_combo_boxes.items()}

        unset = [role for role, ch in role_selections.items() if ch == SELECT_PLACEHOLDER]
        if unset:
            QMessageBox.warning(
                self, "Missing required channel(s)",
                f"You must pick a channel for these roles before continuing:\n{', '.join(unset)}"
            )
            return

        chosen_channels = list(role_selections.values())
        if len(set(chosen_channels)) != len(chosen_channels):
            QMessageBox.warning(
                self, "Duplicate channel",
                "The same channel is assigned to more than one scoring role. "
                "Each scoring role must use a different channel."
            )
            return

        # Remember these choices: channel name -> role (scoring role name,
        # or "Ignore"/"Display Only" for everything else) so future files
        # with the same channel names offer the same choice as a starting
        # point next time.
        for role, ch in role_selections.items():
            self.learned_defaults[ch] = role
        for ch, combo in self.other_combo_boxes.items():
            if ch not in chosen_channels:  # scoring assignment always wins
                self.learned_defaults[ch] = combo.currentText()
        _save_learned_defaults(self.learned_defaults)

        self.accept()

    def get_channel_map(self):
        """
        Returns:
        {
          "scoring": {"EEG1": "ch_name", "EEG2": "ch_name", "EMG": "ch_name"},
          "display_only": ["ch_name", ...],
          "ignore": ["ch_name", ...]
        }
        """
        scoring = {role: combo.currentText() for role, combo in self.role_combo_boxes.items()}
        scoring_channels = set(scoring.values())

        display_only = []
        ignore = []
        for ch, combo in self.other_combo_boxes.items():
            if ch in scoring_channels:
                continue  # already used as a scoring channel, takes priority
            if combo.currentText() == "Display Only":
                display_only.append(ch)
            else:
                ignore.append(ch)

        return {"scoring": scoring, "display_only": display_only, "ignore": ignore}

    def prefill_from_channel_map(self, channel_map):
        """Pre-fill this dialog's dropdowns from a previously-saved channel
        map for this specific file (used when re-opening/editing an
        existing selection, as opposed to the general cross-file learned
        defaults)."""
        for role, ch in channel_map.get("scoring", {}).items():
            combo = self.role_combo_boxes.get(role)
            if combo is not None and ch in self.ch_names:
                combo.setCurrentText(ch)

        for ch in channel_map.get("display_only", []):
            combo = self.other_combo_boxes.get(ch)
            if combo is not None:
                combo.setCurrentText("Display Only")

        for ch in channel_map.get("ignore", []):
            combo = self.other_combo_boxes.get(ch)
            if combo is not None:
                combo.setCurrentText("Ignore")


def channel_map_path(edf_filepath):
    return edf_filepath.replace(".edf", "_channel_map.json")


def save_channel_map(edf_filepath, channel_map):
    map_path = channel_map_path(edf_filepath)
    with open(map_path, "w") as f:
        json.dump(channel_map, f, indent=2)
    return map_path


def load_channel_map(edf_filepath):
    map_path = channel_map_path(edf_filepath)
    if os.path.exists(map_path):
        with open(map_path, "r") as f:
            return json.load(f)
    return None
