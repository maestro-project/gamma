Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 64, Y: 224, X: 224, R: 3, S: 3 }
Dataflow {
TemporalMap(3,3) R;
TemporalMap(12,12) X';
TemporalMap(2,2) S;
SpatialMap(51,51) K;
TemporalMap(28,28) Y';
TemporalMap(40,40) C;
Cluster(80,P);
TemporalMap(1,1) K;
TemporalMap(12,12) X';
TemporalMap(1,1) C;
SpatialMap(1,1) Y';
TemporalMap(1,1) R;
TemporalMap(3,3) S;
}
}
}