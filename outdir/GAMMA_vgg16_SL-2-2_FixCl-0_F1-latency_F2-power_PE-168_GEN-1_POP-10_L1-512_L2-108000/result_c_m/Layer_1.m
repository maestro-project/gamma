Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 3, Y: 224, X: 224, R: 3, S: 3 }
Dataflow {
TemporalMap(2,2) R;
TemporalMap(14,14) K;
TemporalMap(3,3) S;
SpatialMap(216,216) Y';
TemporalMap(3,3) X';
TemporalMap(1,1) C;
Cluster(48,P);
TemporalMap(1,1) X';
TemporalMap(3,3) C;
SpatialMap(1,1) K;
TemporalMap(1,1) S;
TemporalMap(1,1) R;
TemporalMap(1,1) Y';
}
}
}