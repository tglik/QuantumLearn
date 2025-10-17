OPENQASM 2.0;
include "qelib1.inc";
gate unitary q0 { u(pi/2,pi,-pi) q0; }
gate multiplexer_dg q0 { unitary q0; }
gate isometry_to_uncompute_dg q0 { multiplexer_dg q0; }
gate state_preparation(param0,param1) q0 { isometry_to_uncompute_dg q0; }
gate initialize(param0,param1) q0 { reset q0; state_preparation(0.7071067811865475,-0.7071067811865475) q0; }
gate mcx q0,q1,q2,q3,q4 { h q4; cp(pi/2) q3,q4; h q4; h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; h q4; cp(-pi/2) q3,q4; h q4; h q3; t q3; cx q2,q3; tdg q3; h q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; h q3; t q3; cx q2,q3; tdg q3; h q3; h q4; cp(pi/8) q0,q4; h q4; cx q0,q1; h q4; cp(-pi/8) q1,q4; h q4; cx q0,q1; h q4; cp(pi/8) q1,q4; h q4; cx q1,q2; h q4; cp(-pi/8) q2,q4; h q4; cx q0,q2; h q4; cp(pi/8) q2,q4; h q4; cx q1,q2; h q4; cp(-pi/8) q2,q4; h q4; cx q0,q2; h q4; cp(pi/8) q2,q4; h q4; }
gate mcx_2643096872784 q0,q1,q2,q3 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }
gate gate_U__s_ q0,q1,q2,q3 { h q0; h q1; h q2; h q3; x q0; x q1; x q2; x q3; h q3; mcx_2643096872784 q0,q1,q2,q3; h q3; x q0; x q1; x q2; x q3; h q0; h q1; h q2; h q3; }
qreg v[4];
qreg c[4];
qreg out[1];
creg cbits[4];
initialize(0.7071067811865475,-0.7071067811865475) out[0];
h v[0];
x v[3];
barrier v[0],v[1],v[2],v[3],c[0],c[1],c[2],c[3],out[0];
cx v[0],c[0];
cx v[1],c[0];
cx v[0],c[1];
cx v[2],c[1];
cx v[1],c[2];
cx v[3],c[2];
cx v[2],c[3];
cx v[3],c[3];
mcx c[0],c[1],c[2],c[3],out[0];
cx v[0],c[0];
cx v[1],c[0];
cx v[0],c[1];
cx v[2],c[1];
cx v[1],c[2];
cx v[3],c[2];
cx v[2],c[3];
cx v[3],c[3];
barrier v[0],v[1],v[2],v[3],c[0],c[1],c[2],c[3],out[0];
gate_U__s_ v[0],v[1],v[2],v[3];
cx v[0],c[0];
cx v[1],c[0];
cx v[0],c[1];
cx v[2],c[1];
cx v[1],c[2];
cx v[3],c[2];
cx v[2],c[3];
cx v[3],c[3];
mcx c[0],c[1],c[2],c[3],out[0];
cx v[0],c[0];
cx v[1],c[0];
cx v[0],c[1];
cx v[2],c[1];
cx v[1],c[2];
cx v[3],c[2];
cx v[2],c[3];
cx v[3],c[3];
barrier v[0],v[1],v[2],v[3],c[0],c[1],c[2],c[3],out[0];
gate_U__s_ v[0],v[1],v[2],v[3];
measure v[0] -> cbits[0];
measure v[1] -> cbits[1];
measure v[2] -> cbits[2];
measure v[3] -> cbits[3];
