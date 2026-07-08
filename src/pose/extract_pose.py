import cv2
from Pose2Sim import Pose2Sim
from rtmlib import Body, draw_skeleton
import json
import numpy as np

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Vidéo chargée :")
    print(f"  FPS : {fps}")
    print(f"  Résolution : {width}x{height}")
    print(f"  Nombre de frames : {total_frames}")
    
    cap.release()

def show_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (640, 360))
        cv2.imshow('Video', frame_resized)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def estimate_pose_2d(video_path):
    detector = Body(mode='performance', to_openpose=False, backend='onnxruntime', device='cuda')
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        keypoints, scores = detector(frame)
        
        frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.3)

        frame_resized = cv2.resize(frame, (640, 360))
        cv2.imshow("Pose 2D", frame_resized)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def save_keypoints(video_path, output_path):
    detector = Body(mode='performance')

    cap =cv2.VideoCapture(video_path)
    keypoints_list = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = detector(frame)

        keypoints_list.append({
            "frame": frame_idx,
            "keypoints": keypoints.tolist(),
            "scores": scores.tolist()
        })

        frame_idx += 1
        print(f"Processed frame {frame_idx}")

    cap.release()

    with open(output_path, 'w') as f:
        json.dump(keypoints_list, f)
    print(f"Keypoints saved to {output_path}")

if __name__ == "__main__":
    load_video("data/test.mp4")
    # show_video("data/test.mp4")
    # estimate_pose_2d("data/test.mp4")
    save_keypoints("data/test.mp4", "data/keypoints_2d.json")