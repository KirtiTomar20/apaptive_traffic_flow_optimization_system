import cv2
import pandas as pd
import traci
from ultralytics import YOLO
import cvzone
from JobUtils.DynamicObjectTraker import Tracker
from TrafficUtils.TrafficMetrics import TrafficMetricsStore
from JobUtils.SumoController import SumoSimulationManager

vehicles = {}
sumoEnv = SumoSimulationManager()
trafficMetrics = TrafficMetricsStore.Global()


class TrafficCameraProcessor:
    incoming_count = 0

    def __init__(self, video_path, model_path, class_list_path):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.class_list = self.load_class_list(class_list_path)
        self.tracker = Tracker()

        self.cy3 = 275
        self.cy2 = 108  # Line positions
        self.offset = 4  # Line thickness offset

        self.cap = cv2.VideoCapture(video_path)

    def load_class_list(self, class_list_path):
        with open(class_list_path, "r") as file:
            class_list = file.read().split("\n")
        return class_list

    def run_video_processing(self, traci_obj, incoming_count=incoming_count):
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            frame_count += 1
            if not ret:
                break
            if frame_count % 5 == 0:
                frame = cv2.resize(frame, (1020, 500))
            else:
                continue

            # Get predictions from the YOLO model
            results = self.model.predict(frame)
            boxes = results[0].boxes.data
            px = pd.DataFrame(boxes).astype("float")

            detections = []
            class_list = self.load_class_list("coco.txt")
            for index, row in px.iterrows():
                x1, y1, x2, y2, d = map(int, row[[0, 1, 2, 3, 5]])
                if class_list[d] == 'car':  # Adjust if tracking other classes
                    detections.append([x1, y1, x2, y2])

            bbox_ids = self.tracker.update(detections)
            for box_id in bbox_ids:
                x1, y1, x2, y2, id1 = box_id
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                traci_obj.simulationStep()
                # Check and update vehicle state for crossing green line
                if (center_y + self.offset) > self.cy2 > (center_y - self.offset):
                    if id1 not in vehicles or not vehicles[id1].get('crossed_green', False):
                        vehicles[id1] = vehicles.get(id1, {})
                        vehicles[id1]['crossed_green'] = True
                        sumoEnv.add_vehicle_to_N2TL(traci_client=traci_obj, vehicle_id=id1)
                        incoming_count += 1  # Count vehicle as incoming

                # Check and update vehicle state for crossing red line
                if (center_y + self.offset) > self.cy3 > (center_y - self.offset):
                    print("Touched red line")
                    if id1 not in vehicles:
                        print("Id correct")
                        # print(list(vehicles.keys()))
                        for vehicle_id in list(vehicles.keys()):
                            if vehicles[vehicle_id].get('crossed_green'):
                                print(f"Inside red line check: {vehicle_id} has crossed green light.")
                                del vehicles[vehicle_id]
                                incoming_count -= 1  # Decrement count if vehicle crosses both lines

            print(vehicles)
            trafficMetrics.setQueueLength(incoming_count)
            # Draw lines and texts for visualization
            cv2.line(frame, (250, self.cy3), (464, self.cy3), (0, 0, 255), 2)  # Red line
            cv2.line(frame, (147, self.cy2), (208, self.cy2), (0, 255, 0), 2)  # Green line
            cvzone.putTextRect(frame, f'Incoming Vehicle Count: {incoming_count}', (10, 40), scale=1, thickness=4,
                               colorR=(0, 255, 0))
            # cvzone.putTextRect(frame, f'Outgoing: {outgoing_count}', (10, 80), scale=1, thickness=4, colorR=(0,0,255))

            cv2.imshow("Traffic Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    traci2 = traci.connect(port=sumoEnv.port, numRetries=10, label="vehicleAdder")
    traci2.setOrder(2)
    vobj = TrafficCameraProcessor("traffic_intersection_short.mp4", "yolov8s.pt",
                                  "coco.txt")
    vobj.run_video_processing(traci2)
