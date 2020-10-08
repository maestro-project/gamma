Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 3, Y: 224, X: 224, R: 3, S: 3 }
Dataflow {
TemporalMap(20,20) Y';
TemporalMap(121,121) X';
TemporalMap(1,1) S;
SpatialMap(1,1) K;
TemporalMap(1,1) R;
TemporalMap(3,3) C;
Cluster(20,P);
TemporalMap(59,59) K;
TemporalMap(1,1) S;
TemporalMap(1,1) X';
TemporalMap(1,1) R;
TemporalMap(2,2) C;
SpatialMap(1,1) Y';
}
}
}