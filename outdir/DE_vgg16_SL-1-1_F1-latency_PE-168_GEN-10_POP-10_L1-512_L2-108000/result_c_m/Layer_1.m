Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 3, Y: 224, X: 224, R: 3, S: 3 }
Dataflow {
TemporalMap(1,1) K;
TemporalMap(74,74) X';
SpatialMap(1,1) Y';
TemporalMap(1,1) S;
TemporalMap(3,3) C;
TemporalMap(1,1) R;
}
}
}