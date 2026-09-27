import socket

import state_manager
import events
import json

import time
import traceback

class SocketListener:
    def __init__(self):
        self.has_received: bool = False
        self.buffer_size: int = 1024 * 1024
        self.should_run = True

    def run(self, port_num: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('127.0.0.1', port_num))
        sock.settimeout(0.5)
        print("Created socket on port {}, listening...".format(port_num))
        while self.should_run:
            try:
                data, addr = sock.recvfrom(self.buffer_size)
            except:
                continue

            has_received: True

            try:
                j = json.loads(data.decode("utf-8"))
            except json.decoder.JSONDecodeError as err:
                print("ERROR parsing received text to JSON:", err)

                view_range = 10
                start, stop = max(0, err.pos - view_range), min(err.pos + view_range, len(err.doc) - 1)
                snippet = err.doc[start:stop].replace('\r', '').replace('\n', ' ')
                snippet_prefix = "Received JSON: "
                underline = (' ' * (len(snippet)//2 + len(snippet_prefix))) + '^ HERE'
                print("\t" + snippet_prefix + snippet)
                print("\t" + underline)
                j = None

            if not (j is None):
                # The State-Set Editor flags its stream so only IT enables the pose keys (B/V).
                # Checked BEFORE the edit-mode skip so B still works to exit while editing.
                if isinstance(j, dict) and j.get("allow_pose_edit"):
                    state_manager.pose_edit_time = time.time()

                # In pose-edit mode the visualizer owns the scene (keyboard posing) — ignore
                # incoming gamestates so they don't overwrite the edits.
                if getattr(state_manager, "edit_mode", False):
                    continue

                # PLAY driver HUD (live reward / record status), shown in the overlay by render().
                state_manager.hud_text = j.get("hud", "") if isinstance(j, dict) else ""
                # Optional camera lock to a specific car (streaming tools, e.g. view_credit.py POV on
                # the highest-credit player). None -> leave the user's spectate choice alone.
                state_manager.forced_spectate_idx = j.get("spectate_idx") if isinstance(j, dict) else None
                # A play/test tool capturing input -> suspend RocketSimVis's own camera keybinds.
                if isinstance(j, dict) and j.get("capture_input"):
                    state_manager.input_capture_time = time.time()

                # Optional GAIL-training discriminator readout: {"p_ai": float|null, "label": str,
                # "hard": bool}. Drawn by main.py's render_pai_hud() ONLY when this is non-None —
                # its presence in the packet IS the flag, so normal (non-GAIL) senders that omit
                # "gail_hud" leave this None and the overlay stays off automatically.
                state_manager.gail_hud = j.get("gail_hud") if isinstance(j, dict) else None

                # Optional match scoreboard: {"score_blue": int, "score_orange": int,
                # "time_left": float, "is_overtime": bool}. Same flag-gate as gail_hud — its
                # presence IS the flag, so senders that omit "scoreboard" leave this None and
                # render_scoreboard_hud() draws nothing.
                state_manager.scoreboard = j.get("scoreboard") if isinstance(j, dict) else None

                # Optional answer from the sender to a visualizer key (1/2/3, S/D), shown in the panel.
                if isinstance(j, dict) and j.get("vis_msg"):
                    state_manager.vis_msg = (str(j["vis_msg"]), time.time())

                recv_time = time.time()

                # Queued for smooth playback (state_manager.PlayoutClock): the render thread applies it
                # at its playback time, so uneven arrival no longer shows as hitches. read_from_json still
                # runs in order on the one live state object (prev must be the immediately previous packet).
                if isinstance(j, dict):
                    s_play = state_manager.playout_clock.schedule(recv_time)
                    state_manager.queue_packet(s_play, j)
                    # Reconstruct RL game events (jumps, flips, flip resets, touches, demos...) for sounds
                    # and effects by diffing against the previous packet, timed on the playback clock.
                    events.g_detector.process(j, state_manager.global_state_manager.state.boost_pad_locations, s_play)
                    state_manager.record_packet(recv_time, data)


    def stop_async(self):
        self.should_run = False