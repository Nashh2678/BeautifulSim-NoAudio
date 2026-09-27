# Networking format

The gamestate should be sent as JSON to the renderer over its UDP socket.
The default port is `9273` (set `RSV_PORT` to change it). Packets can arrive at any rate; the visualizer
interpolates between them.

Some JSON fields are optional.

The format for the gamestate, in pseudo-JSON, is:

```
# Physics state
{ 
	"pos": [ <x>, <y>, <z> ],
	
	OPTIONAL "forward": [ <x>, <y>, <z> ], # Forward direction as a normalized vector
	OPTIONAL "up": [ <x>, <y>, <z> ], # Upward direction as a normalized vector
	# NOTE: If rotation ("forward" and "up") are not provided, 
	#       the visualizer will track its own internal rotation for the object, 
	#       and update it with "ang_vel" every frame
	
	"vel": [ <x>, <y>, <z> ],
	"ang_vel": [ <x>, <y>, <z> ]
}
```

```
{
	"ball_phys": <physics state>,
	
	"cars": [
		{ # Example car
			"team_num": <0 or 1>, # Blue = 0, orange = 1
			
			"phys": <physics state>,
			
			# The input that PRODUCED this state (e.g. GigaLearnCPP's player.prevAction). A new "jump"
			# press is shown as a jump / flip / double jump even if the car is back on a surface by the
			# next packet (wall dashes); "boost" drives the boost flame.
			OPTIONAL "controls": { 
				"throttle": <v>, "steer": <v>, 
				"pitch": <v>, "yaw": <v>, "roll": <v>, 
				"boost": <bool>, "jump": <bool>, "handbrake": <bool>, 
			},
			
			"boost_amount": <v> (from 0 to 100),
			"on_ground": <bool>,
			OPTIONAL "has_flip": <bool>, # a flip / double jump is still available (exact flip + flip-reset detection)
			OPTIONAL "has_flipped_or_double_jumped": <bool>, # older alternative to "has_flip"
			OPTIONAL "ball_touched": <bool>, # the car touched the ball during this step
			"is_demoed": <bool>
		}
	],

	# Goal replay pause: while "ball_hidden" is true the ball isn't drawn and the camera looks at "pos"
	OPTIONAL "ball_hidden": <bool>,
	OPTIONAL "goal_celebration": { "pos": [ <x>, <y>, <z> ], "time_left": <seconds> },

	# Lock the camera to one car (index into "cars")
	OPTIONAL "spectate_idx": <int>,
	
	# Use this to change the locations and number of boost pads
	# If you never set this, the visualizer will use the soccar boost locations with RLGym/RLBot ordering
	OPTIONAL "boost_pad_locations": [ [<x>, <y>, <z>], [<x>, <y>, <z>], ... ],
	
	# If you don't provide this, no boost pads will be rendered
	OPTIONAL "boost_pad_states": [ <bool>, <bool>, ... ],

	# Answer to a visualizer request (see below), shown in the top-left panel for a few seconds.
	# Send it in ONE packet, then leave it out again.
	OPTIONAL "vis_msg": <string>
}
```

## Visualizer -> sender side channel (optional)

The visualizer also sends short UDP datagrams to `127.0.0.1:9275`. A sender that doesn't bind that port
never sees them and needs no changes.

| datagram | meaning |
|---|---|
| `1` / `0` | the visualizer window gained / lost focus (sent periodically). A sender can pause while it is `0`. |
| `view:1`, `view:2`, `view:3` | key 1/2/3: the viewer wants to watch 1v1 / 2v2 / 3v3 |
| `det:0`, `det:1` | key S/D: the viewer wants stochastic / deterministic bot actions |

The sender decides what to do, for example switch only to team sizes its bot's observation supports.
It answers with `"vis_msg"` in its next packet (e.g. `"Spectating 2v2"` or
`"3v3: not supported by this bot's observation"`). If no answer comes within 1.5 s the visualizer
shows that the sender doesn't handle these keys.
