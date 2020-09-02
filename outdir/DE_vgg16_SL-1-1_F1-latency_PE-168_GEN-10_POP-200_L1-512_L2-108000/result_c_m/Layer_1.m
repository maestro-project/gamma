Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 3, Y: 224, X: 224, R: 3, S: 3 }
Dataflow {
TemporalMap(3,3) C;
TemporalMap(140,140) X';
TemporalMap(1,1) R;
TemporalMap(5,5) K;
TemporalMap(1,1) S;
SpatialMap(42,42) Y';
}
}
}