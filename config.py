SENSOR_FL = 2
SENSOR_FR = 3

CAR = 0
LARGE_VEHICLE = 1
TRUCK = 2
BUS = 3
TRAIN = 4
BICYCLE = 5
MOTORIZED_TWO_WHEELER = 6
PEDESTRIAN = 7
PEDESTRIAN_GROUP = 8
ANIMAL = 9
OTHER = 10
STATIC = 11

labels_map = {
    TRUCK: LARGE_VEHICLE,
    BUS: LARGE_VEHICLE,
    MOTORIZED_TWO_WHEELER: BICYCLE,
    PEDESTRIAN_GROUP: PEDESTRIAN,
}
drop_labels = [OTHER, ANIMAL, TRAIN]
final_labels = [CAR, LARGE_VEHICLE, BICYCLE, PEDESTRIAN, STATIC]
label_to_index = {label: idx for idx, label in enumerate(final_labels)}
index_to_label = {idx: label for idx, label in enumerate(final_labels)}
N_LABELS = len(final_labels)
