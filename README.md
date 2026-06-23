# hailo-apps

`hailo_apps` runs Hailo detection models on Raspberry Pi camera frames and uses
the results to control pan-and-tilt servos.

## Components

The application stack builds from Hailo inference to camera capture and
rotation control. `FaceTracker` combines these layers, while `Servos` provides
the hardware output used by `RotatorApp`.

```text
┌─────────────────────────────────────────────────────────────────────┐
│  HAILO_APPS // FACE TRACKING CONTROL LOOP                          │
└─────────────────────────────────────────────────────────────────────┘

 [ RASPBERRY PI CAMERA ]
            │
            │  FRAME_STREAM
            ▼
 ┌─────────────────────┐
 │      PicamApp       │
 │   CAPTURE :: LOOP   │
 └──────────┬──────────┘
            │  on_frame(frame)
            ▼
 ┌─────────────────────┐                 ┌─────────────────────┐
 │     RotatorApp      │                 │      HailoApp       │
 │  TRACK :: CONTROL   │                 │   MODEL :: SETUP    │
 └──────────┬──────────┘                 └──────────┬──────────┘
            │  get_centroid(frame)                  │ initializes
            ▼                                       ▼
 ┌─────────────────────┐   inference    ┌─────────────────────┐
 │     FaceTracker     │ ─────────────► │     Hailo model     │
 │  FACE :: CENTROID   │ ◄───────────── │   HEF :: DETECT     │
 └──────────┬──────────┘   detections   └─────────────────────┘
            │  centroid (x, y)
            ▼
 ┌─────────────────────┐    servo angles    ┌─────────────────────┐
 │  ANGLE CALCULATION  │ ─────────────────► │       Servos        │
 └─────────────────────┘                    │  PAN :: TILT        │
                                            └──────────┬──────────┘
                                                       │
                      ◄────── PHYSICAL FEEDBACK ───────┘
```

### [HailoApp](src/hailo_apps/meta/interfaces/hailo_app.py)

`HailoApp` resolves a local or remote HEF model, downloads supported models when
necessary, and initializes Hailo inference.

### [PicamApp](src/hailo_apps/meta/interfaces/picam_app.py)

`PicamApp` configures a Raspberry Pi camera, captures frames in a background
thread, and passes them to the application for processing. It uses the
inference initialized by `HailoApp`.

### [RotatorApp](src/hailo_apps/meta/interfaces/rotator_app.py)

`RotatorApp` converts the position of a detected object into bounded
pan-and-tilt servo movements. It processes the frames supplied by `PicamApp`
and sends the resulting angles to `Servos`.

### [Servos](src/hailo_apps/servos/servos.py)

`Servos` controls two channels through Adafruit ServoKit: channel 0 for X and
channel 1 for Y.

### [FaceTracker](src/hailo_apps/apps/face_tracker.py)

`FaceTracker` runs face detection on each camera frame, calculates the center of
the first detected face, and moves the pan-and-tilt servos to keep that face at
the desired position in the frame. It is the concrete application built from
`HailoApp`, `PicamApp`, `RotatorApp`, and `Servos`.
