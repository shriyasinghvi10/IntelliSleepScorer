import json
import os

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QScrollArea, QWidget, QMessageBox
)


# Role choices shown per-channel, depending on which model will be used to score.
ROLE_OPTIONS_2EEG = ["Ignore", "Display Only", "EEG1 (scoring)", "EEG2 (scoring)", "EMG (scoring)"]
ROLE_OPTIONS_1EEG = ["Ignore", "Display Only", "EEG (scoring)", "EMG (scoring)"]

# The exact roles each model needs assigned before it can extract features.
REQUIRED_ROLES = {
    "1_LightGBM-2EEG": ["EEG1 (scoring)", "EEG2 (scoring)", "EMG (scoring)"],
    "2_LightGBM-1EEG": ["EEG (scoring)", "EMG (scoring)"],
}


class ChannelSelectDialog(QDialog):
    """
    Shown once per EDF file (and cached to disk afterward). Lets the user
    assign a role to every channel found in the file:

      - One of the "(scoring)" roles: this channel is fed into feature
        extraction / the LightGBM model.
      - "Display Only": shown in the viewer (e.g. a TTL/laser/optogenetic
        channel) but never used for scoring.
      - "Ignore": not loaded at all.
    """

    def __init__(self, ch_names, model_name, filename="", parent=None):
        super().__init__(parent)
        self.ch_names = ch_names
        self.model_name = model_name
        self.role_options = ROLE_OPTIONS_2EEG if model_name == "1_LightGBM-2EEG" else ROLE_OPTIONS_1EEG
        self.required_roles = REQUIRED_ROLES.get(model_name, [])
        self.combo_boxes = {}

        self.setWindowTitle(f"Select Channels - {filename}")
        self.setMinimumWidth(480)
        self.setMinimumHeight(400)

        outer_layout = QVBoxLayout(self)

        header = QLabel(
            f"Model '{model_name}' needs: {', '.join(self.required_roles)}.\n"
            "Assign a role to every channel below. Channels marked 'Display Only' "
            "will be shown in the viewer but not used for scoring. Channels marked "
            "'Ignore' will not be loaded at all."
        )
        header.setWordWrap(True)
        outer_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        form_layout = QVBoxLayout(scroll_content)

        for ch in ch_names:
            row = QHBoxLayout()
            label = QLabel(ch)
            label.setMinimumWidth(220)
            combo = QComboBox()
            combo.addItems(self.role_options)
            guessed = self._guess_role(ch)
            if guessed:
                combo.setCurrentText(guessed)
            row.addWidget(label)
            row.addWidget(combo)
            form_layout.addLayout(row)
            self.combo_boxes[ch] = combo

        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll)

        self.button_ok = QPushButton("Confirm Channel Selection")
        self.button_ok.clicked.connect(self.validate_and_accept)
        outer_layout.addWidget(self.button_ok)

    def _guess_role(self, ch_name):
        """Best-effort guess from the channel name, so the user usually
        just has to confirm rather than pick from scratch every time."""
        name_lower = ch_name.lower()
        if "emg" in name_lower:
            return "EMG (scoring)" if "EMG (scoring)" in self.role_options else None
        if "eeg1" in name_lower or "eeg_1" in name_lower:
            return "EEG1 (scoring)" if "EEG1 (scoring)" in self.role_options else "EEG (scoring)"
        if "eeg2" in name_lower or "eeg_2" in name_lower:
            return "EEG2 (scoring)" if "EEG2 (scoring)" in self.role_options else None
        if "eeg" in name_lower:
            return "EEG (scoring)" if "EEG (scoring)" in self.role_options else "EEG1 (scoring)"
        if any(tag in name_lower for tag in ["ttl", "laser", "opto", "stim", "trig"]):
            return "Display Only"
        return None

    def validate_and_accept(self):
        selected_roles = [combo.currentText() for combo in self.combo_boxes.values()]

        missing = [r for r in self.required_roles if r not in selected_roles]
        if missing:
            QMessageBox.warning(
                self, "Missing required channel(s)",
                f"You must assign these roles before continuing:\n{', '.join(missing)}"
            )
            return

        for role in self.required_roles:
            if selected_roles.count(role) > 1:
                QMessageBox.warning(
                    self, "Duplicate role",
                    f"Role '{role}' is assigned to more than one channel. "
                    "Each scoring role must be assigned to exactly one channel."
                )
                return

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
        scoring = {}
        display_only = []
        ignore = []
        for ch, combo in self.combo_boxes.items():
            role = combo.currentText()
            if role == "Ignore":
                ignore.append(ch)
            elif role == "Display Only":
                display_only.append(ch)
            else:
                clean_role = role.split(" (")[0]  # "EEG1 (scoring)" -> "EEG1"
                scoring[clean_role] = ch
        return {"scoring": scoring, "display_only": display_only, "ignore": ignore}


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
