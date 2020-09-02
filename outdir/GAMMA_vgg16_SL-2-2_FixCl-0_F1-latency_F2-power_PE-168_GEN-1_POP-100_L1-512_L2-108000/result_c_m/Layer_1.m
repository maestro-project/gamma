Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 3, Y: 224, X: 224, R: 3, S: 3 }
Dataflow {
TemporalMap(43,43) K;
TemporalMap(1,1) C;
TemporalMap(2,2) R;
TemporalMap(2,2) S;
SpatialMap(23,23) Y';
TemporalMap(3,3) X';
Cluster(43,P);
TemporalMap(1,1) C;
TemporalMap(1,1) X';
TemporalMap(1,1) S;
TemporalMap(1,1) Y';
SpatialMap(1,1) K;
TemporalMap(1,1) R;
}
}
}