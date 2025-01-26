from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shredathletics.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize MediaPipe Pose with more robust settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

@app.get("/health")
async def health_check():
    """Endpoint to verify server is running."""
    return {"status": "healthy"}

def normalize_angle(angle):
    """Normalize angle to handle different camera perspectives."""
    return min(max(angle, 0), 180)

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle between three points with normalization."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return normalize_angle(angle)

def analyse_running_form(video_path: str) -> dict:
    """Analyse running form using BlazePose with improved consistency."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError("Could not open video file")

        frame_count = 0
        valid_frame_count = 0
        analysis_results = {
            "stride_length": [],
            "knee_angle": [],
            "hip_angle": [],
            "arm_swing": [],
            "posture_alignment": []
        }
        
        # Track temporal consistency
        prev_landmarks = None
        movement_threshold = 0.1
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Check landmark visibility and confidence
                key_points = [
                    mp_pose.PoseLandmark.LEFT_HIP,
                    mp_pose.PoseLandmark.LEFT_KNEE,
                    mp_pose.PoseLandmark.LEFT_ANKLE,
                    mp_pose.PoseLandmark.RIGHT_HIP,
                    mp_pose.PoseLandmark.RIGHT_KNEE,
                    mp_pose.PoseLandmark.RIGHT_ANKLE
                ]
                
                if all(landmarks[point.value].visibility > 0.7 for point in key_points):
                    valid_frame_count += 1
                    
                    # Calculate normalized stride length
                    hip_width = np.linalg.norm(
                        np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]) -
                        np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
                    )
                    
                    # Normalize stride length by hip width for scale invariance
                    if landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > 0.7:
                        left_ankle = np.array([
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                        ])
                        normalized_stride = left_ankle[0] / hip_width
                        analysis_results["stride_length"].append(normalized_stride)
                    
                    # Calculate bilateral knee angles
                    for side in ["LEFT", "RIGHT"]:
                        hip = np.array([
                            landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x,
                            landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y
                        ])
                        knee = np.array([
                            landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x,
                            landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].y
                        ])
                        ankle = np.array([
                            landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x,
                            landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].y
                        ])
                        
                        knee_angle = calculate_angle(hip, knee, ankle)
                        analysis_results["knee_angle"].append(knee_angle)
                    
                    # Check temporal consistency
                    if prev_landmarks is not None:
                        movement = np.mean([abs(landmarks[i].x - prev_landmarks[i].x) for i in range(33)])
                        if movement > movement_threshold:
                            analysis_results["posture_alignment"].append(1.0)
                        else:
                            analysis_results["posture_alignment"].append(0.0)
                    
                    prev_landmarks = landmarks
            
            frame_count += 1
            if frame_count > 300:  # Analyse max 10 seconds at 30fps
                break
        
        cap.release()
        
        if valid_frame_count < 10:
            logger.error("Insufficient valid frames for analysis")
            raise ValueError("Not enough valid frames for accurate analysis")

        # Process collected data with confidence weighting
        return {
            "stride_length": process_metric(analysis_results["stride_length"]),
            "knee_angle": process_metric(analysis_results["knee_angle"]),
            "posture_alignment": process_metric(analysis_results["posture_alignment"]),
            "overall_score": calculate_overall_score(analysis_results),
            "recommendations": generate_recommendations(analysis_results),
            "confidence_score": min(1.0, valid_frame_count / frame_count) * 100
        }
    except Exception as e:
        logger.error(f"Error in analyse_running_form: {str(e)}")
        raise

def process_metric(values: List[float]) -> dict:
    """Process collected metrics with improved statistical analysis."""
    if not values:
        return {"average": 0, "consistency": 0, "confidence": 0}
    
    # Remove outliers using IQR method
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    valid_values = [x for x in values if (q1 - 1.5 * iqr) <= x <= (q3 + 1.5 * iqr)]
    
    if not valid_values:
        return {"average": np.mean(values), "consistency": 0, "confidence": 0}
    
    return {
        "average": np.mean(valid_values),
        "consistency": 100 - (np.std(valid_values) * 100),
        "confidence": len(valid_values) / len(values) * 100
    }

def calculate_overall_score(results: dict) -> int:
    """Calculate overall running form score with confidence weighting."""
    scores = []
    weights = []
    
    # Score stride length consistency
    if results["stride_length"]:
        stride_stats = process_metric(results["stride_length"])
        stride_score = stride_stats["consistency"]
        scores.append(stride_score)
        weights.append(stride_stats["confidence"] / 100)
    
    # Score knee angle
    if results["knee_angle"]:
        knee_stats = process_metric(results["knee_angle"])
        optimal_knee_angle = 90
        knee_angles = results["knee_angle"]
        avg_knee_angle = np.mean(knee_angles)
        knee_score = 100 - min(100, abs(avg_knee_angle - optimal_knee_angle))
        scores.append(knee_score)
        weights.append(knee_stats["confidence"] / 100)
    
    # Score posture alignment
    if results["posture_alignment"]:
        posture_stats = process_metric(results["posture_alignment"])
        scores.append(posture_stats["consistency"])
        weights.append(posture_stats["confidence"] / 100)
    
    # Calculate weighted average
    if not scores or not weights:
        return 70
    
    weighted_score = np.average(scores, weights=weights)
    return int(weighted_score)

def generate_recommendations(results: dict) -> List[str]:
    """Generate detailed recommendations based on analysis with confidence thresholds."""
    recommendations = []
    
    # Only provide recommendations if we have sufficient confidence
    confidence_threshold = 70  # Minimum confidence level to make recommendations
    
    # Analyse stride length
    if results["stride_length"]:
        stride_stats = process_metric(results["stride_length"])
        if stride_stats["confidence"] > confidence_threshold:
            consistency = stride_stats["consistency"]
            if consistency < 70:
                recommendations.append(
                    f"Your stride length varies significantly (consistency: {consistency:.1f}%). " +
                    "Focus on maintaining a more consistent stride length by using a metronome app or counting your steps."
                )
            elif consistency < 85:
                recommendations.append(
                    f"Your stride length shows moderate consistency (consistency: {consistency:.1f}%). " +
                    "Continue working on maintaining an even stride pattern."
                )
            else:
                recommendations.append(
                    f"Excellent stride consistency at {consistency:.1f}%! " +
                    "This helps prevent energy waste and reduces injury risk."
                )

    # Analyse knee angle
    if results["knee_angle"]:
        knee_stats = process_metric(results["knee_angle"])
        if knee_stats["confidence"] > confidence_threshold:
            avg_knee_angle = knee_stats["average"]
            if avg_knee_angle < 85:
                recommendations.append(
                    f"Your knee angle (currently {avg_knee_angle:.1f}°) is too acute. " +
                    "Focus on lifting your knees higher during your stride to achieve closer to 90°. " +
                    "This will improve your power output and reduce stress on your joints."
                )
            elif avg_knee_angle > 95:
                recommendations.append(
                    f"Your knee angle (currently {avg_knee_angle:.1f}°) is too obtuse. " +
                    "Try shortening your stride slightly to achieve a more optimal knee bend around 90°. " +
                    "This will improve your running efficiency."
                )
            else:
                recommendations.append(
                    f"Great knee angle at {avg_knee_angle:.1f}°! " +
                    "This is in the optimal range for efficient running."
                )

    # Analyse posture alignment
    if results["posture_alignment"]:
        posture_stats = process_metric(results["posture_alignment"])
        if posture_stats["confidence"] > confidence_threshold:
            consistency = posture_stats["consistency"]
            if consistency < 80:
                recommendations.append(
                    "Your running posture shows significant variation. " +
                    "Focus on maintaining a stable core and keeping your upper body straight. " +
                    "Consider core strengthening exercises to improve stability."
                )
            else:
                recommendations.append(
                    "Good posture stability! Continue maintaining an upright position " +
                    "with your shoulders relaxed and core engaged."
                )

    # Add general form tips if we don't have enough confidence in specific measurements
    if len(recommendations) < 2:
        recommendations.extend([
            "Focus on landing mid-foot rather than heel-striking to reduce impact and improve efficiency.",
            "Keep your head level and look about 20 feet ahead while running.",
            "Maintain a slight forward lean from the ankles, not the waist.",
            "Consider recording yourself from different angles to track improvements in your form."
        ])

    return recommendations

@app.post("/analyse-running")
async def analyse_running(video: UploadFile = File(...)):
    """Endpoint to analyse running form from uploaded video."""
    try:
        logger.info(f"Received video upload: {video.filename}")
        
        # Verify file type
        if not video.filename.lower().endswith(('.mp4', '.mov', '.avi')):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid file type. Please upload MP4, MOV, or AVI video."}
            )

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            logger.info(f"Saving video to temporary file: {temp_video.name}")
            content = await video.read()
            temp_video.write(content)
            temp_video_path = temp_video.name

        try:
            # Analyse the video
            logger.info("Starting video analysis")
            results = analyse_running_form(temp_video_path)
            logger.info("Analysis completed successfully")
            return JSONResponse(content=results)
        
        finally:
            # Clean up
            logger.info("Cleaning up temporary file")
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing video: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 