Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 3, Y: 224, X: 224, R: 3, S: 3 }
Dataflow {
TemporalMap(2,2) Y';
SpatialMap(1,1) C;
TemporalMap(1,1) R;
TemporalMap(122,122) X';
TemporalMap(57,57) K;
TemporalMap(1,1) S;
Cluster(122,P);
TemporalMap(1,1) S;
SpatialMap(1,1) X';
TemporalMap(3,3) R;
TemporalMap(1,1) Y';
TemporalMap(54,54) K;
TemporalMap(1,1) C;
}
}
}