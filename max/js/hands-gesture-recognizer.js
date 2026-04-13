(async () => {
  const vision = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js");
  const { GestureRecognizer, FilesetResolver, DrawingUtils } = vision;

  const video = document.getElementById('videoel');
  const overlay = document.getElementById('overlay');
  const canvas = overlay.getContext('2d');

  let gestureRecognizer;
  let camera;
  let runningMode = "VIDEO";
  let lastVideoTime = -1;
  let results = undefined;
  const drawingUtils = new DrawingUtils(canvas);

  // only inlet you need — camera on/off
  window.max.bindInlet('camera_toggle', async function (enable) {
    if (enable) {
      if (camera) return; // already running
      try {
        video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
        runningMode = "IMAGE"; // force mode switch
        await setRunningMode("VIDEO");
      } catch (e) {
        window.max.outlet("error", `Camera restart failed: ${e.message}`);
      }
    } else {
      stopCamera();
      canvas.clearRect(0, 0, overlay.width, overlay.height);
    }
  });

  function stopCamera() {
    // stop the mediapipe camera loop
    if (camera) {
      camera.stop();
      camera = undefined;
    }
    // stop all video tracks
    if (video.srcObject) {
      video.srcObject.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
    }
    // force stop the video element
    video.pause();
    video.src = "";
    video.load();
    // clear canvas
    canvas.clearRect(0, 0, overlay.width, overlay.height);
  }

  const startVideo = () => {
    camera = new Camera(video, {
      onFrame: async () => {
        if (video && runningMode === "VIDEO") {
          let nowInMs = Date.now();
          if (lastVideoTime !== video.currentTime) {
            lastVideoTime = video.currentTime;
            results = gestureRecognizer.recognizeForVideo(video, nowInMs);
            results.image = video;
            onResultsHands(results);
          }
        }
      },
      width: 640,
      height: 480
    });
    camera.start();
  }

  const setRunningMode = async (running_mode) => {
    if (running_mode === runningMode) return;
    canvas.clearRect(0, 0, overlay.width, overlay.height);
    runningMode = running_mode;
    await gestureRecognizer.setOptions({ runningMode: running_mode });
    if (running_mode === "VIDEO") startVideo();
  };

  function onResultsHands(results) {
    canvas.save();
    canvas.clearRect(0, 0, overlay.width, overlay.height);

    // draw camera feed
    canvas.drawImage(results.image, 0, 0, overlay.width, overlay.height);

    const output = {};

    if (results.handednesses) {
      for (const hand of results.handednesses) {
        Object.values(HAND_LANDMARKS).forEach(([landmark, index]) => {
          try {
            const handIndex = results.handednesses.length > 1 ? Number(hand[0].index) : 0;
            // flip left/right since camera is mirrored
            const handName = hand[0].categoryName === "Right" ? "Left" : "Right";
            output[handName] = output[handName] || {};
            output[handName][landmark] = results.landmarks[handIndex][index];
            if (results.gestures.length > 0) {
              const categoryName = results.gestures[handIndex][0].categoryName;
              output[handName]["Gestures"] = output[handName]["Gestures"] || {};
              output[handName]["Gestures"][categoryName] = results.gestures[handIndex][0].score;
            }
          } catch (e) {
            console.error(e);
          }
        });
      }
    }

    // draw hand skeleton
    if (results.landmarks) {
      for (const landmarks of results.landmarks) {
        drawingUtils.drawConnectors(landmarks, GestureRecognizer.HAND_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 1
        });
        drawingUtils.drawLandmarks(landmarks, {
          color: "#FF0000",
          fillColor: '#FF0000',
          lineWidth: (data) => 1 + data.from.z * -2,
          radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, .1, 2, 1)
        });
      }
    }

    window.max.setDict('hands_landmarkdict', output);
    window.max.outlet("update");
    canvas.restore();
  }

  // load mediapipe model
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  gestureRecognizer = await GestureRecognizer.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numHands: 2
  });

  // start camera automatically
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
    startVideo();
    window.max.outlet("ready");
  } catch (e) {
    window.max.outlet("error", `Camera failed: ${e.message}`);
  }

})();