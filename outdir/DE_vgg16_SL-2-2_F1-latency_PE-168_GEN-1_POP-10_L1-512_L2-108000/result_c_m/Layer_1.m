Network 0 {
Layer CONV {
Type: CONV
Dimensions { K: 64, C: 3, Y: 224, X: 224, R: 3, S: 3 }
Dataflow {
TemporalMap(1,1) R;
TemporalMap(1,1) S;
TemporalMap(150,150) X';
TemporalMap(2,2) C;
TemporalMap(20,20) K;
SpatialMap(88,88) Y';
Cluster(2,P);
TemporalMap(1,1) R;
TemporalMap(25,25) K;
TemporalMap(1,1) S;
TemporalMap(65,65) X';
TemporalMap(122,122) Y';
SpatialMap(2,2) C;
}
}
}