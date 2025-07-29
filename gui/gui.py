import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel

class TressetteWindow(QWidget):
    def __init__(self, engine, ai_player):
        super().__init__()
        self.engine = engine
        self.ai_player = ai_player
        self.setWindowTitle("Tressette")
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()

        self.status = QLabel("Welcome to Tressette!")
        self.layout.addWidget(self.status)

        self.play_button = QPushButton("Play AI Turn")
        self.play_button.clicked.connect(self.play_turn)
        self.layout.addWidget(self.play_button)

        self.setLayout(self.layout)

    def play_turn(self):
        if self.engine.current_player == self.ai_player.player_index:
            obs = self.engine._get_obs()
            player = self.engine.players[self.engine.current_player]
            valid = self.engine.get_valid_actions()
            action = self.ai_player.act(obs, player, valid)
            self.engine.step(action)
            self.status.setText(f"AI played action: {action}")

def launch_game(engine, ai_player):
    app = QApplication(sys.argv)
    window = TressetteWindow(engine, ai_player)
    window.show()
    sys.exit(app.exec_())