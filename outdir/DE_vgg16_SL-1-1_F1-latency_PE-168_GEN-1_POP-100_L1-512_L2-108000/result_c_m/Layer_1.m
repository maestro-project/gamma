Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 3, Y: 224, X: 224, R: 3, S: 3 }
Dataflow {
TemporalMap(2,2) C;
SpatialMap(23,23) X';
TemporalMap(1,1) S;
TemporalMap(155,155) Y';
TemporalMap(25,25) K;
TemporalMap(1,1) R;
}
}
}