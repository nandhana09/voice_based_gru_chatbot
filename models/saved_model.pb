±/
Ķ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18µ°-

embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	č*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	č*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:6*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:6*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:6*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:6*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*$
shared_namegru/gru_cell/kernel
{
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes

:`*
dtype0

gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*.
shared_namegru/gru_cell/recurrent_kernel

1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes

: `*
dtype0
~
gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*"
shared_namegru/gru_cell/bias
w
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes

:`*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	č*,
shared_nameAdam/embedding/embeddings/m

/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	č*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

: *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:6*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:6*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:6*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:6*
dtype0

Adam/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*+
shared_nameAdam/gru/gru_cell/kernel/m

.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/m*
_output_shapes

:`*
dtype0
¤
$Adam/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/m

8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/m*
_output_shapes

: `*
dtype0

Adam/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*)
shared_nameAdam/gru/gru_cell/bias/m

,Adam/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/m*
_output_shapes

:`*
dtype0

Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	č*,
shared_nameAdam/embedding/embeddings/v

/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	č*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

: *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:6*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:6*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:6*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:6*
dtype0

Adam/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*+
shared_nameAdam/gru/gru_cell/kernel/v

.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/v*
_output_shapes

:`*
dtype0
¤
$Adam/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/v

8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/v*
_output_shapes

: `*
dtype0

Adam/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*)
shared_nameAdam/gru/gru_cell/bias/v

,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/v*
_output_shapes

:`*
dtype0

NoOpNoOp
Ī3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*3
value’2Bü2 Bõ2

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
l
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
Š
'iter

(beta_1

)beta_2
	*decay
+learning_ratembmcmd!me"mf,mg-mh.mivjvkvl!vm"vn,vo-vp.vq
 
8
0
,1
-2
.3
4
5
!6
"7
8
0
,1
-2
.3
4
5
!6
"7
­
/layer_regularization_losses
regularization_losses
0non_trainable_variables
trainable_variables
1metrics

2layers
3layer_metrics
		variables
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
4layer_regularization_losses
regularization_losses
5non_trainable_variables
trainable_variables
6metrics

7layers
8layer_metrics
	variables
~

,kernel
-recurrent_kernel
.bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
 
 

,0
-1
.2

,0
-1
.2
¹
=layer_regularization_losses
regularization_losses
>non_trainable_variables
trainable_variables
?metrics

@layers
Alayer_metrics
	variables

Bstates
 
 
 
­
Clayer_regularization_losses
regularization_losses
Dnon_trainable_variables
trainable_variables
Emetrics

Flayers
Glayer_metrics
	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Hlayer_regularization_losses
regularization_losses
Inon_trainable_variables
trainable_variables
Jmetrics

Klayers
Llayer_metrics
	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
­
Mlayer_regularization_losses
#regularization_losses
Nnon_trainable_variables
$trainable_variables
Ometrics

Players
Qlayer_metrics
%	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEgru/gru_cell/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEgru/gru_cell/recurrent_kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEgru/gru_cell/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 

R0
S1
#
0
1
2
3
4
 
 
 
 
 
 
 

,0
-1
.2

,0
-1
.2
­
Tlayer_regularization_losses
9regularization_losses
Unon_trainable_variables
:trainable_variables
Vmetrics

Wlayers
Xlayer_metrics
;	variables
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ytotal
	Zcount
[	variables
\	keras_api
D
	]total
	^count
_
_fn_kwargs
`	variables
a	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

[	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

`	variables

VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/gru/gru_cell/kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/gru/gru_cell/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/gru/gru_cell/kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/gru/gru_cell/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_embedding_inputPlaceholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
Ž
StatefulPartitionedCallStatefulPartitionedCallserving_default_embedding_inputembedding/embeddingsgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_14244
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ć
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOp,Adam/gru/gru_cell/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOp,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_16897
Ź
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biastotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/gru/gru_cell/kernel/m$Adam/gru/gru_cell/recurrent_kernel/mAdam/gru/gru_cell/bias/mAdam/embedding/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/gru/gru_cell/kernel/v$Adam/gru/gru_cell/recurrent_kernel/vAdam/gru/gru_cell/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_17006ę¢,
½Æ

8__inference___backward_gpu_gru_with_fallback_14499_14635
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŅ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationß
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationö
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape©
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapesų
õ:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_f85ad4e4-9dae-4161-82a3-d19a54787a62*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_14634*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ :1-
+
_output_shapes
:’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :1-
+
_output_shapes
:’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
½Æ

8__inference___backward_gpu_gru_with_fallback_13860_13996
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŅ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationß
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationö
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape©
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapesų
õ:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_41f76db4-6a90-4306-bc36-84a0308e6d2b*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_13995*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ :1-
+
_output_shapes
:’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :1-
+
_output_shapes
:’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
½Æ

8__inference___backward_gpu_gru_with_fallback_13473_13609
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŅ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationß
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationö
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape©
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapesų
õ:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_0e9f412c-411a-49ad-b27d-6154347bf343*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_13608*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ :1-
+
_output_shapes
:’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :1-
+
_output_shapes
:’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
Ų
while_cond_16381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_16381___redundant_placeholder03
/while_while_cond_16381___redundant_placeholder13
/while_while_cond_16381___redundant_placeholder23
/while_while_cond_16381___redundant_placeholder33
/while_while_cond_16381___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
äG
Ę
%__forward_gpu_gru_with_fallback_12770

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cć
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_a5576c00-a82c-4209-912f-0f034a2c0696*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_12635_12771*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
¾=
¹
'__inference_gpu_gru_with_fallback_15756

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cÖ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_fc00a0e9-20f3-41f2-8dfc-07e7197849b8*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
	
Ų
while_cond_14327
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_14327___redundant_placeholder03
/while_while_cond_14327___redundant_placeholder13
/while_while_cond_14327___redundant_placeholder23
/while_while_cond_14327___redundant_placeholder33
/while_while_cond_14327___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:

ū
E__inference_sequential_layer_call_and_return_conditional_losses_14120
embedding_input
embedding_14098
	gru_14101
	gru_14103
	gru_14105
dense_14109
dense_14111
dense_1_14114
dense_1_14116
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!embedding/StatefulPartitionedCall¢gru/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_14098*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_132132#
!embedding/StatefulPartitionedCall­
gru/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0	gru_14101	gru_14103	gru_14105*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_139982
gru/StatefulPartitionedCall”
(global_average_pooling1d/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_140332*
(global_average_pooling1d/PartitionedCall­
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_14109dense_14111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_140512
dense/StatefulPartitionedCall¬
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14114dense_1_14116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_140782!
dense_1/StatefulPartitionedCall
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:X T
'
_output_shapes
:’’’’’’’’’
)
_user_specified_nameembedding_input
	
Ų
while_cond_14736
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_14736___redundant_placeholder03
/while_while_cond_14736___redundant_placeholder13
/while_while_cond_14736___redundant_placeholder23
/while_while_cond_14736___redundant_placeholder33
/while_while_cond_14736___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:

Ė
>__inference_gru_layer_call_and_return_conditional_losses_16691
inputs_0 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
zeros
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:`*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity_2§
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:’’’’’’’’’ :’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_164732
PartitionedCall}

Identity_3IdentityPartitionedCall:output:1*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’::::^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs/0
Ł
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_14033

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’ :S O
+
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
«

#__inference_gru_layer_call_fn_16702
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_127732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs/0
Æ
Ŗ
B__inference_dense_1_layer_call_and_return_conditional_losses_14078

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:6*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’62
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’62	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’62	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
·G
Ę
%__forward_gpu_gru_with_fallback_11589

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cŚ
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_694dec2c-2806-406a-9bce-6f01fde30db6*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_11454_11590*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Ź2
į
while_body_13302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
äG
Ę
%__forward_gpu_gru_with_fallback_16301

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cć
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_9d1ce7b3-c0b8-4fd3-8825-8b7d76e40dfe*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_16166_16302*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
E
¢
__inference_standard_gru_14419

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_14328*
condR
while_cond_14327*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_f85ad4e4-9dae-4161-82a3-d19a54787a62*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
	
Ų
while_cond_12861
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_12861___redundant_placeholder03
/while_while_cond_12861___redundant_placeholder13
/while_while_cond_12861___redundant_placeholder23
/while_while_cond_12861___redundant_placeholder33
/while_while_cond_12861___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ø
T
8__inference_global_average_pooling1d_layer_call_fn_16724

inputs
identityŃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_140332
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’ :S O
+
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs

Ł
*__inference_sequential_layer_call_fn_15104

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĆ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_141942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ä
o
)__inference_embedding_layer_call_fn_15121

inputs
unknown
identity¢StatefulPartitionedCallė
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_132132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ģ
É
>__inference_gru_layer_call_and_return_conditional_losses_13611

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
zeros
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:`*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity_2
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_133932
PartitionedCallt

Identity_3IdentityPartitionedCall:output:1*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:’’’’’’’’’::::S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
E
¢
__inference_standard_gru_15290

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_15199*
condR
while_cond_15198*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_b6765986-e022-4cb1-9852-b9c581a21f2a*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Ł
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_16719

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’ :S O
+
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
·G
Ę
%__forward_gpu_gru_with_fallback_15505

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cŚ
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_b6765986-e022-4cb1-9852-b9c581a21f2a*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_15370_15506*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
¾=
¹
'__inference_gpu_gru_with_fallback_14498

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cÖ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_f85ad4e4-9dae-4161-82a3-d19a54787a62*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
³°

8__inference___backward_gpu_gru_with_fallback_12635_12771
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŪ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationč
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation’
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape²
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’ :’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_a5576c00-a82c-4209-912f-0f034a2c0696*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_12770*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ ::6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: ::6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ū
#__inference_signature_wrapper_14244
embedding_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_116082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:’’’’’’’’’
)
_user_specified_nameembedding_input
E
¢
__inference_standard_gru_13780

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_13689*
condR
while_cond_13688*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_41f76db4-6a90-4306-bc36-84a0308e6d2b*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
·G
Ę
%__forward_gpu_gru_with_fallback_15043

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cŚ
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_da00b7ed-e318-4c70-b819-7a9f30f07ada*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_14908_15044*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias


#__inference_gru_layer_call_fn_15917

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_139982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
·2
®
E__inference_sequential_layer_call_and_return_conditional_losses_14653

inputs$
 embedding_embedding_lookup_14248$
 gru_read_readvariableop_resource&
"gru_read_1_readvariableop_resource&
"gru_read_2_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityq
embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’2
embedding/Cast’
embedding/embedding_lookupResourceGather embedding_embedding_lookup_14248embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/14248*+
_output_shapes
:’’’’’’’’’*
dtype02
embedding/embedding_lookupę
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/14248*+
_output_shapes
:’’’’’’’’’2%
#embedding/embedding_lookup/Identity¾
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2'
%embedding/embedding_lookup/Identity_1t
	gru/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2ś
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/packed/1
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	gru/zeros
gru/Read/ReadVariableOpReadVariableOp gru_read_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/Read/ReadVariableOpr
gru/IdentityIdentitygru/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:`2
gru/Identity
gru/Read_1/ReadVariableOpReadVariableOp"gru_read_1_readvariableop_resource*
_output_shapes

: `*
dtype02
gru/Read_1/ReadVariableOpx
gru/Identity_1Identity!gru/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2
gru/Identity_1
gru/Read_2/ReadVariableOpReadVariableOp"gru_read_2_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/Read_2/ReadVariableOpx
gru/Identity_2Identity!gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2
gru/Identity_2Ü
gru/PartitionedCallPartitionedCall.embedding/embedding_lookup/Identity_1:output:0gru/zeros:output:0gru/Identity:output:0gru/Identity_1:output:0gru/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_144192
gru/PartitionedCall¤
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesŠ
global_average_pooling1d/MeanMeangru/PartitionedCall:output:18global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
global_average_pooling1d/Mean
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp„
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

dense/Relu„
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:6*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’62
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02 
dense_1/BiasAdd/ReadVariableOp”
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’62
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’62
dense_1/Softmaxm
IdentityIdentitydense_1/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’:::::::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ŃE
¢
__inference_standard_gru_12555

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_12464*
condR
while_cond_12463*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeń
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_a5576c00-a82c-4209-912f-0f034a2c0696*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
·G
Ę
%__forward_gpu_gru_with_fallback_13995

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cŚ
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_41f76db4-6a90-4306-bc36-84a0308e6d2b*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_13860_13996*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
ģ
É
>__inference_gru_layer_call_and_return_conditional_losses_13998

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
zeros
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:`*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity_2
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_137802
PartitionedCallt

Identity_3IdentityPartitionedCall:output:1*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:’’’’’’’’’::::S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
E
¢
__inference_standard_gru_15677

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_15586*
condR
while_cond_15585*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_fc00a0e9-20f3-41f2-8dfc-07e7197849b8*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
¾=
¹
'__inference_gpu_gru_with_fallback_15369

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cÖ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_b6765986-e022-4cb1-9852-b9c581a21f2a*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Ų
|
'__inference_dense_1_layer_call_fn_16775

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallņ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_140782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
·G
Ę
%__forward_gpu_gru_with_fallback_14634

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cŚ
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_f85ad4e4-9dae-4161-82a3-d19a54787a62*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_14499_14635*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
³°

8__inference___backward_gpu_gru_with_fallback_16166_16302
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŪ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationč
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation’
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape²
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’ :’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_9d1ce7b3-c0b8-4fd3-8825-8b7d76e40dfe*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_16301*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ ::6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: ::6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¾=
¹
'__inference_gpu_gru_with_fallback_14907

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cÖ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_da00b7ed-e318-4c70-b819-7a9f30f07ada*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Ź2
į
while_body_15586
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
ŃE
¢
__inference_standard_gru_12953

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_12862*
condR
while_cond_12861*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeń
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_83883df0-de40-4590-a30e-0197ba9ba8d1*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
E
¢
__inference_standard_gru_14828

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_14737*
condR
while_cond_14736*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_da00b7ed-e318-4c70-b819-7a9f30f07ada*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
„
Ø
@__inference_dense_layer_call_and_return_conditional_losses_14051

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
·2
®
E__inference_sequential_layer_call_and_return_conditional_losses_15062

inputs$
 embedding_embedding_lookup_14657$
 gru_read_readvariableop_resource&
"gru_read_1_readvariableop_resource&
"gru_read_2_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityq
embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’2
embedding/Cast’
embedding/embedding_lookupResourceGather embedding_embedding_lookup_14657embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/14657*+
_output_shapes
:’’’’’’’’’*
dtype02
embedding/embedding_lookupę
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/14657*+
_output_shapes
:’’’’’’’’’2%
#embedding/embedding_lookup/Identity¾
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2'
%embedding/embedding_lookup/Identity_1t
	gru/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2ś
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/packed/1
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	gru/zeros
gru/Read/ReadVariableOpReadVariableOp gru_read_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/Read/ReadVariableOpr
gru/IdentityIdentitygru/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:`2
gru/Identity
gru/Read_1/ReadVariableOpReadVariableOp"gru_read_1_readvariableop_resource*
_output_shapes

: `*
dtype02
gru/Read_1/ReadVariableOpx
gru/Identity_1Identity!gru/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2
gru/Identity_1
gru/Read_2/ReadVariableOpReadVariableOp"gru_read_2_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/Read_2/ReadVariableOpx
gru/Identity_2Identity!gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2
gru/Identity_2Ü
gru/PartitionedCallPartitionedCall.embedding/embedding_lookup/Identity_1:output:0gru/zeros:output:0gru/Identity:output:0gru/Identity_1:output:0gru/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_148282
gru/PartitionedCall¤
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesŠ
global_average_pooling1d/MeanMeangru/PartitionedCall:output:18global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
global_average_pooling1d/Mean
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp„
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

dense/Relu„
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:6*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’62
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02 
dense_1/BiasAdd/ReadVariableOp”
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’62
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’62
dense_1/Softmaxm
IdentityIdentitydense_1/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’:::::::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ź2
į
while_body_16382
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
ģ
É
>__inference_gru_layer_call_and_return_conditional_losses_15895

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
zeros
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:`*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity_2
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_156772
PartitionedCallt

Identity_3IdentityPartitionedCall:output:1*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:’’’’’’’’’::::S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
³°

8__inference___backward_gpu_gru_with_fallback_13033_13169
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŪ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationč
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation’
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape²
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’ :’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_83883df0-de40-4590-a30e-0197ba9ba8d1*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_13168*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ ::6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: ::6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¾=
¹
'__inference_gpu_gru_with_fallback_13859

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cÖ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_41f76db4-6a90-4306-bc36-84a0308e6d2b*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
·G
Ę
%__forward_gpu_gru_with_fallback_15892

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cŚ
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_fc00a0e9-20f3-41f2-8dfc-07e7197849b8*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_15757_15893*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
õ=
¹
'__inference_gpu_gru_with_fallback_13032

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cß
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_83883df0-de40-4590-a30e-0197ba9ba8d1*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
õ
ņ
E__inference_sequential_layer_call_and_return_conditional_losses_14148

inputs
embedding_14126
	gru_14129
	gru_14131
	gru_14133
dense_14137
dense_14139
dense_1_14142
dense_1_14144
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!embedding/StatefulPartitionedCall¢gru/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_14126*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_132132#
!embedding/StatefulPartitionedCall­
gru/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0	gru_14129	gru_14131	gru_14133*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_136112
gru/StatefulPartitionedCall”
(global_average_pooling1d/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_140332*
(global_average_pooling1d/PartitionedCall­
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_14137dense_14139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_140512
dense/StatefulPartitionedCall¬
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14142dense_1_14144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_140782!
dense_1/StatefulPartitionedCall
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
³°

8__inference___backward_gpu_gru_with_fallback_16553_16689
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŪ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationč
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation’
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape²
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’ :’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_e9b274cf-1ddd-41d4-81c6-6b67c6ad909d*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_16688*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ ::6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: ::6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
’H
ž
__inference__traced_save_16897
file_prefix3
/savev2_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_m_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_v_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_72db0cc2a9f54c409bef0c85ac6712b0/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameØ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*ŗ
value°B­"B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesĢ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesź
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop5savev2_adam_gru_gru_cell_kernel_m_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop3savev2_adam_gru_gru_cell_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop5savev2_adam_gru_gru_cell_kernel_v_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop3savev2_adam_gru_gru_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesō
ń: :	č: ::6:6: : : : : :`: `:`: : : : :	č: ::6:6:`: `:`:	č: ::6:6:`: `:`: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	č:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:6: 

_output_shapes
:6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :$ 

_output_shapes

:`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	č:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:6: 

_output_shapes
:6:$ 

_output_shapes

:`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:%!

_output_shapes
:	č:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:6: 

_output_shapes
:6:$ 

_output_shapes

:`:$  

_output_shapes

: `:$! 

_output_shapes

:`:"

_output_shapes
: 
®

D__inference_embedding_layer_call_and_return_conditional_losses_15114

inputs
embedding_lookup_15108
identity]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’2
CastĶ
embedding_lookupResourceGatherembedding_lookup_15108Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/15108*+
_output_shapes
:’’’’’’’’’*
dtype02
embedding_lookup¾
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/15108*+
_output_shapes
:’’’’’’’’’2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
®

D__inference_embedding_layer_call_and_return_conditional_losses_13213

inputs
embedding_lookup_13207
identity]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’2
CastĶ
embedding_lookupResourceGatherembedding_lookup_13207Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/13207*+
_output_shapes
:’’’’’’’’’*
dtype02
embedding_lookup¾
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/13207*+
_output_shapes
:’’’’’’’’’2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0**
_input_shapes
:’’’’’’’’’::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ź2
į
while_body_15995
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
E
¢
__inference_standard_gru_11374

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_11283*
condR
while_cond_11282*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_694dec2c-2806-406a-9bce-6f01fde30db6*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Ź2
į
while_body_15199
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
Ź2
į
while_body_14328
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
Æ
Ŗ
B__inference_dense_1_layer_call_and_return_conditional_losses_16766

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:6*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’62
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’62	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’62	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

ū
E__inference_sequential_layer_call_and_return_conditional_losses_14095
embedding_input
embedding_13222
	gru_14021
	gru_14023
	gru_14025
dense_14062
dense_14064
dense_1_14089
dense_1_14091
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!embedding/StatefulPartitionedCall¢gru/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_13222*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_132132#
!embedding/StatefulPartitionedCall­
gru/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0	gru_14021	gru_14023	gru_14025*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_136112
gru/StatefulPartitionedCall”
(global_average_pooling1d/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_140332*
(global_average_pooling1d/PartitionedCall­
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_14062dense_14064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_140512
dense/StatefulPartitionedCall¬
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14089dense_1_14091*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_140782!
dense_1/StatefulPartitionedCall
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:X T
'
_output_shapes
:’’’’’’’’’
)
_user_specified_nameembedding_input
½Æ

8__inference___backward_gpu_gru_with_fallback_15370_15506
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŅ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationß
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationö
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape©
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapesų
õ:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_b6765986-e022-4cb1-9852-b9c581a21f2a*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_15505*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ :1-
+
_output_shapes
:’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :1-
+
_output_shapes
:’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
Ų
while_cond_15585
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_15585___redundant_placeholder03
/while_while_cond_15585___redundant_placeholder13
/while_while_cond_15585___redundant_placeholder23
/while_while_cond_15585___redundant_placeholder33
/while_while_cond_15585___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
	
Ų
while_cond_13301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_13301___redundant_placeholder03
/while_while_cond_13301___redundant_placeholder13
/while_while_cond_13301___redundant_placeholder23
/while_while_cond_13301___redundant_placeholder33
/while_while_cond_13301___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
„
Ø
@__inference_dense_layer_call_and_return_conditional_losses_16746

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ :::O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs

o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13196

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'’’’’’’’’’’’’’’’’’’’’’’’’’’’:e a
=
_output_shapes+
):'’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ō
z
%__inference_dense_layer_call_fn_16755

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallš
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_140512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
	
Ų
while_cond_12463
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_12463___redundant_placeholder03
/while_while_cond_12463___redundant_placeholder13
/while_while_cond_12463___redundant_placeholder23
/while_while_cond_12463___redundant_placeholder33
/while_while_cond_12463___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
	
Ų
while_cond_15198
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_15198___redundant_placeholder03
/while_while_cond_15198___redundant_placeholder13
/while_while_cond_15198___redundant_placeholder23
/while_while_cond_15198___redundant_placeholder33
/while_while_cond_15198___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
«

#__inference_gru_layer_call_fn_16713
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_131712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs/0
Ź2
į
while_body_13689
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
	
Ų
while_cond_15994
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_15994___redundant_placeholder03
/while_while_cond_15994___redundant_placeholder13
/while_while_cond_15994___redundant_placeholder23
/while_while_cond_15994___redundant_placeholder33
/while_while_cond_15994___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ź2
į
while_body_14737
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
½Æ

8__inference___backward_gpu_gru_with_fallback_11454_11590
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŅ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationß
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationö
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape©
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapesų
õ:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_694dec2c-2806-406a-9bce-6f01fde30db6*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_11589*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ :1-
+
_output_shapes
:’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :1-
+
_output_shapes
:’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¾=
¹
'__inference_gpu_gru_with_fallback_11453

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cÖ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_694dec2c-2806-406a-9bce-6f01fde30db6*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
ŃE
¢
__inference_standard_gru_16473

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_16382*
condR
while_cond_16381*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeń
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_e9b274cf-1ddd-41d4-81c6-6b67c6ad909d*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias

o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_16730

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'’’’’’’’’’’’’’’’’’’’’’’’’’’’:e a
=
_output_shapes+
):'’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs


#__inference_gru_layer_call_fn_15906

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_136112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
­=
ź
 __inference__wrapped_model_11608
embedding_input/
+sequential_embedding_embedding_lookup_11203/
+sequential_gru_read_readvariableop_resource1
-sequential_gru_read_1_readvariableop_resource1
-sequential_gru_read_2_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identity
sequential/embedding/CastCastembedding_input*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’2
sequential/embedding/Cast¶
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_11203sequential/embedding/Cast:y:0*
Tindices0*>
_class4
20loc:@sequential/embedding/embedding_lookup/11203*+
_output_shapes
:’’’’’’’’’*
dtype02'
%sequential/embedding/embedding_lookup
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/11203*+
_output_shapes
:’’’’’’’’’20
.sequential/embedding/embedding_lookup/Identityß
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:’’’’’’’’’22
0sequential/embedding/embedding_lookup/Identity_1
sequential/gru/ShapeShape9sequential/embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
sequential/gru/Shape
"sequential/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/gru/strided_slice/stack
$sequential/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_1
$sequential/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_2¼
sequential/gru/strided_sliceStridedSlicesequential/gru/Shape:output:0+sequential/gru/strided_slice/stack:output:0-sequential/gru/strided_slice/stack_1:output:0-sequential/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/gru/strided_slicez
sequential/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/gru/zeros/mul/yØ
sequential/gru/zeros/mulMul%sequential/gru/strided_slice:output:0#sequential/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/mul}
sequential/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
sequential/gru/zeros/Less/y£
sequential/gru/zeros/LessLesssequential/gru/zeros/mul:z:0$sequential/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/Less
sequential/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
sequential/gru/zeros/packed/1æ
sequential/gru/zeros/packedPack%sequential/gru/strided_slice:output:0&sequential/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/gru/zeros/packed}
sequential/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/gru/zeros/Const±
sequential/gru/zerosFill$sequential/gru/zeros/packed:output:0#sequential/gru/zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
sequential/gru/zeros“
"sequential/gru/Read/ReadVariableOpReadVariableOp+sequential_gru_read_readvariableop_resource*
_output_shapes

:`*
dtype02$
"sequential/gru/Read/ReadVariableOp
sequential/gru/IdentityIdentity*sequential/gru/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:`2
sequential/gru/Identityŗ
$sequential/gru/Read_1/ReadVariableOpReadVariableOp-sequential_gru_read_1_readvariableop_resource*
_output_shapes

: `*
dtype02&
$sequential/gru/Read_1/ReadVariableOp
sequential/gru/Identity_1Identity,sequential/gru/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2
sequential/gru/Identity_1ŗ
$sequential/gru/Read_2/ReadVariableOpReadVariableOp-sequential_gru_read_2_readvariableop_resource*
_output_shapes

:`*
dtype02&
$sequential/gru/Read_2/ReadVariableOp
sequential/gru/Identity_2Identity,sequential/gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2
sequential/gru/Identity_2©
sequential/gru/PartitionedCallPartitionedCall9sequential/embedding/embedding_lookup/Identity_1:output:0sequential/gru/zeros:output:0 sequential/gru/Identity:output:0"sequential/gru/Identity_1:output:0"sequential/gru/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_113742 
sequential/gru/PartitionedCallŗ
:sequential/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential/global_average_pooling1d/Mean/reduction_indicesü
(sequential/global_average_pooling1d/MeanMean'sequential/gru/PartitionedCall:output:1Csequential/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2*
(sequential/global_average_pooling1d/MeanĄ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&sequential/dense/MatMul/ReadVariableOpŃ
sequential/dense/MatMulMatMul1sequential/global_average_pooling1d/Mean:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential/dense/MatMulæ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential/dense/BiasAdd
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential/dense/ReluĘ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:6*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpÉ
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’62
sequential/dense_1/MatMulÅ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpĶ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’62
sequential/dense_1/BiasAdd
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’62
sequential/dense_1/Softmaxx
IdentityIdentity$sequential/dense_1/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’:::::::::X T
'
_output_shapes
:’’’’’’’’’
)
_user_specified_nameembedding_input
õ=
¹
'__inference_gpu_gru_with_fallback_12634

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cß
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_a5576c00-a82c-4209-912f-0f034a2c0696*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
½Æ

8__inference___backward_gpu_gru_with_fallback_14908_15044
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŅ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationß
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationö
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape©
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapesų
õ:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_da00b7ed-e318-4c70-b819-7a9f30f07ada*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_15043*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ :1-
+
_output_shapes
:’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :1-
+
_output_shapes
:’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
E
¢
__inference_standard_gru_13393

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_13302*
condR
while_cond_13301*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeč
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm„
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_0e9f412c-411a-49ad-b27d-6154347bf343*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
·G
Ę
%__forward_gpu_gru_with_fallback_13608

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cŚ
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_0e9f412c-411a-49ad-b27d-6154347bf343*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_13473_13609*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
ŗ
ā
*__inference_sequential_layer_call_fn_14213
embedding_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĢ
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_141942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:’’’’’’’’’
)
_user_specified_nameembedding_input

É
>__inference_gru_layer_call_and_return_conditional_losses_13171

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
zeros
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:`*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity_2„
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:’’’’’’’’’ :’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_129532
PartitionedCall}

Identity_3IdentityPartitionedCall:output:1*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’::::\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ŃE
¢
__inference_standard_gru_16086

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3X
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ü
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_1n
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:’’’’’’’’’`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:’’’’’’’’’`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
splitj
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:’’’’’’’’’`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:’’’’’’’’’`2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim“
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
Tanh\
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`* 
_read_only_resource_inputs
 *
bodyR
while_body_15995*
condR
while_cond_15994*R
output_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’    22
0TensorArrayV2Stack/TensorListStack/element_shapeń
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_9d1ce7b3-c0b8-4fd3-8825-8b7d76e40dfe*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias

Ł
*__inference_sequential_layer_call_fn_15083

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĆ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_141482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
äG
Ę
%__forward_gpu_gru_with_fallback_16688

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cć
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_e9b274cf-1ddd-41d4-81c6-6b67c6ad909d*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_16553_16689*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias

É
>__inference_gru_layer_call_and_return_conditional_losses_12773

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
zeros
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:`*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity_2„
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:’’’’’’’’’ :’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_125552
PartitionedCall}

Identity_3IdentityPartitionedCall:output:1*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’::::\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ź2
į
while_body_12862
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
ŗ
ā
*__inference_sequential_layer_call_fn_14167
embedding_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallĢ
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_141482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:’’’’’’’’’
)
_user_specified_nameembedding_input
äG
Ę
%__forward_gpu_gru_with_fallback_13168

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cć
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_83883df0-de40-4590-a30e-0197ba9ba8d1*
api_preferred_deviceGPU*T
backward_function_name:8__inference___backward_gpu_gru_with_fallback_13033_13169*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
	
Ų
while_cond_13688
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_13688___redundant_placeholder03
/while_while_cond_13688___redundant_placeholder13
/while_while_cond_13688___redundant_placeholder23
/while_while_cond_13688___redundant_placeholder33
/while_while_cond_13688___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
	
Ų
while_cond_11282
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_11282___redundant_placeholder03
/while_while_cond_11282___redundant_placeholder13
/while_while_cond_11282___redundant_placeholder23
/while_while_cond_11282___redundant_placeholder33
/while_while_cond_11282___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*D
_input_shapes3
1: : : : :’’’’’’’’’ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ģ
É
>__inference_gru_layer_call_and_return_conditional_losses_15508

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
zeros
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:`*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity_2
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_152902
PartitionedCallt

Identity_3IdentityPartitionedCall:output:1*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:’’’’’’’’’::::S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
õ=
¹
'__inference_gpu_gru_with_fallback_16165

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cß
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_9d1ce7b3-c0b8-4fd3-8825-8b7d76e40dfe*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
½Æ

8__inference___backward_gpu_gru_with_fallback_15757_15893
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape½
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’25
3gradients/strided_slice_grad/StridedSliceGrad/begin°
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endø
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesŅ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationß
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĘ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:’’’’’’’’’ 2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:’’’’’’’’’:’’’’’’’’’ : :Ą%*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationö
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:’’’’’’’’’2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeź
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: 2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: 2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: 2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_1_grad/ShapeĒ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_2_grad/ShapeÉ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       2 
gradients/Reshape_3_grad/ShapeÉ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_4_grad/ShapeÉ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_5_grad/ShapeÉ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        2 
gradients/Reshape_6_grad/ShapeÉ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_7_grad/ShapeÅ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_8_grad/ShapeÅ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2 
gradients/Reshape_9_grad/ShapeÅ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: 2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_10_grad/ShapeČ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_11_grad/ShapeÉ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: 2!
gradients/Reshape_12_grad/ShapeÉ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationß
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationß
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationß
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: 2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationß
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationß
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationß
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ą2
gradients/split_2_grad/concat£
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:`2
gradients/split_grad/concat«
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   2
gradients/Reshape_grad/ShapeĆ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`2 
gradients/Reshape_grad/Reshape©
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:’’’’’’’’’2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_2 

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

: `2

Identity_3”

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes

:`2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapesų
õ:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ : :’’’’’’’’’ ::’’’’’’’’’ : ::’’’’’’’’’:’’’’’’’’’ : :Ą%::’’’’’’’’’ : ::::::: : : *<
api_implements*(gru_fc00a0e9-20f3-41f2-8dfc-07e7197849b8*
api_preferred_deviceGPU*@
forward_function_name'%__forward_gpu_gru_with_fallback_15892*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:’’’’’’’’’ :1-
+
_output_shapes
:’’’’’’’’’ :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :1-
+
_output_shapes
:’’’’’’’’’ : 

_output_shapes
::1-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:’’’’’’’’’:1
-
+
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :!

_output_shapes	
:Ą%: 

_output_shapes
::-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ė
>__inference_gru_layer_call_and_return_conditional_losses_16304
inputs_0 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
zeros
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:`*
dtype02
Read/ReadVariableOpf
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype02
Read_1/ReadVariableOpl

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype02
Read_2/ReadVariableOpl

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`2

Identity_2§
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:’’’’’’’’’ :’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference_standard_gru_160862
PartitionedCall}

Identity_3IdentityPartitionedCall:output:1*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:’’’’’’’’’’’’’’’’’’::::^ Z
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
"
_user_specified_name
inputs/0
õ
ņ
E__inference_sequential_layer_call_and_return_conditional_losses_14194

inputs
embedding_14172
	gru_14175
	gru_14177
	gru_14179
dense_14183
dense_14185
dense_1_14188
dense_1_14190
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!embedding/StatefulPartitionedCall¢gru/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_14172*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_132132#
!embedding/StatefulPartitionedCall­
gru/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0	gru_14175	gru_14177	gru_14179*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_139982
gru/StatefulPartitionedCall”
(global_average_pooling1d/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_140332*
(global_average_pooling1d/PartitionedCall­
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_14183dense_14185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_140512
dense/StatefulPartitionedCall¬
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_14188dense_1_14190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_140782!
dense_1/StatefulPartitionedCall
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’62

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ź2
į
while_body_12464
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
½
²
!__inference__traced_restore_17006
file_prefix)
%assignvariableop_embedding_embeddings#
assignvariableop_1_dense_kernel!
assignvariableop_2_dense_bias%
!assignvariableop_3_dense_1_kernel#
assignvariableop_4_dense_1_bias 
assignvariableop_5_adam_iter"
assignvariableop_6_adam_beta_1"
assignvariableop_7_adam_beta_2!
assignvariableop_8_adam_decay)
%assignvariableop_9_adam_learning_rate+
'assignvariableop_10_gru_gru_cell_kernel5
1assignvariableop_11_gru_gru_cell_recurrent_kernel)
%assignvariableop_12_gru_gru_cell_bias
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_13
/assignvariableop_17_adam_embedding_embeddings_m+
'assignvariableop_18_adam_dense_kernel_m)
%assignvariableop_19_adam_dense_bias_m-
)assignvariableop_20_adam_dense_1_kernel_m+
'assignvariableop_21_adam_dense_1_bias_m2
.assignvariableop_22_adam_gru_gru_cell_kernel_m<
8assignvariableop_23_adam_gru_gru_cell_recurrent_kernel_m0
,assignvariableop_24_adam_gru_gru_cell_bias_m3
/assignvariableop_25_adam_embedding_embeddings_v+
'assignvariableop_26_adam_dense_kernel_v)
%assignvariableop_27_adam_dense_bias_v-
)assignvariableop_28_adam_dense_1_kernel_v+
'assignvariableop_29_adam_dense_1_bias_v2
.assignvariableop_30_adam_gru_gru_cell_kernel_v<
8assignvariableop_31_adam_gru_gru_cell_recurrent_kernel_v0
,assignvariableop_32_adam_gru_gru_cell_bias_v
identity_34¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9®
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*ŗ
value°B­"B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesŅ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesŲ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¤
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¢
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¤
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5”
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¢
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ŗ
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Æ
AssignVariableOp_10AssignVariableOp'assignvariableop_10_gru_gru_cell_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¹
AssignVariableOp_11AssignVariableOp1assignvariableop_11_gru_gru_cell_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12­
AssignVariableOp_12AssignVariableOp%assignvariableop_12_gru_gru_cell_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13”
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14”
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15£
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16£
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17·
AssignVariableOp_17AssignVariableOp/assignvariableop_17_adam_embedding_embeddings_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Æ
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19­
AssignVariableOp_19AssignVariableOp%assignvariableop_19_adam_dense_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20±
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Æ
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_1_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¶
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_gru_gru_cell_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ą
AssignVariableOp_23AssignVariableOp8assignvariableop_23_adam_gru_gru_cell_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24“
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_gru_gru_cell_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25·
AssignVariableOp_25AssignVariableOp/assignvariableop_25_adam_embedding_embeddings_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Æ
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27­
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_dense_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_1_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Æ
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_1_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¶
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_gru_gru_cell_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ą
AssignVariableOp_31AssignVariableOp8assignvariableop_31_adam_gru_gru_cell_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32“
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_gru_gru_cell_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp“
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33§
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*
_input_shapes
: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
õ=
¹
'__inference_gpu_gru_with_fallback_16552

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cß
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:’’’’’’’’’’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*d
_input_shapesS
Q:’’’’’’’’’’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_e9b274cf-1ddd-41d4-81c6-6b67c6ad909d*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
¾=
¹
'__inference_gpu_gru_with_fallback_13472

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2

ExpandDimsP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
: : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:Ą2	
ReshapeT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: 2
transpose_1j
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: 2
transpose_2j
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 2
transpose_3j
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm}
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  2
transpose_4j
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  2
transpose_5j
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  2
transpose_6j
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:2
	Reshape_6j
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
: 2
	Reshape_9l

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_11l

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
: 2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisŌ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:Ą%2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cÖ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:’’’’’’’’’ :’’’’’’’’’ : :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ö
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:’’’’’’’’’ *
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/perm
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2
transpose_7{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:’’’’’’’’’ *
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identityv

Identity_1Identitytranspose_7:y:0	^CudnnRNN*
T0*+
_output_shapes
:’’’’’’’’’ 2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_2b

Identity_3Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*[
_input_shapesJ
H:’’’’’’’’’:’’’’’’’’’ :`: `:`*<
api_implements*(gru_0e9f412c-411a-49ad-b27d-6154347bf343*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinit_h:FB

_output_shapes

:`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
ī
T
8__inference_global_average_pooling1d_layer_call_fn_16735

inputs
identityŚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_131962
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'’’’’’’’’’’’’’’’’’’’’’’’’’’’:e a
=
_output_shapes+
):'’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ź2
į
while_body_11283
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstackĆ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:’’’’’’’’’*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem”
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimÄ
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/MatMul_1
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:’’’’’’’’’`2
while/BiasAdd_1`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dimĢ
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:’’’’’’’’’ :’’’’’’’’’ :’’’’’’’’’ *
	num_split2
while/split_1
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/addj
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_1p
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Sigmoid_1|
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/mulz
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_2c

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

while/Tanh{
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/sub/xx
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
	while/subr
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/mul_2w
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/add_3Ó
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ø
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*R
_input_shapesA
?: : : : :’’’’’’’’’ : : :`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:’’’’’’’’’ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`"øL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ŗ
serving_default¦
K
embedding_input8
!serving_default_embedding_input:0’’’’’’’’’;
dense_10
StatefulPartitionedCall:0’’’’’’’’’6tensorflow/serving/predict:øŚ
¹0
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
*r&call_and_return_all_conditional_losses
s_default_save_signature
t__call__"Å-
_tf_keras_sequential¦-{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "dtype": "float32", "input_dim": 1000, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 20}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 54, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "dtype": "float32", "input_dim": 1000, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 20}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 54, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
¤

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
*u&call_and_return_all_conditional_losses
v__call__"
_tf_keras_layerė{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "dtype": "float32", "input_dim": 1000, "output_dim": 16, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
°
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"

_tf_keras_rnn_layeré	{"class_name": "GRU", "name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 16]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 16]}}

regularization_losses
trainable_variables
	variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"
_tf_keras_layerź{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ģ

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
*{&call_and_return_all_conditional_losses
|__call__"Ē
_tf_keras_layer­{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
ó

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
*}&call_and_return_all_conditional_losses
~__call__"Ī
_tf_keras_layer“{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 54, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
ć
'iter

(beta_1

)beta_2
	*decay
+learning_ratembmcmd!me"mf,mg-mh.mivjvkvl!vm"vn,vo-vp.vq"
	optimizer
 "
trackable_list_wrapper
X
0
,1
-2
.3
4
5
!6
"7"
trackable_list_wrapper
X
0
,1
-2
.3
4
5
!6
"7"
trackable_list_wrapper
Ź
/layer_regularization_losses
regularization_losses
0non_trainable_variables
trainable_variables
1metrics

2layers
3layer_metrics
		variables
t__call__
s_default_save_signature
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
':%	č2embedding/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
4layer_regularization_losses
regularization_losses
5non_trainable_variables
trainable_variables
6metrics

7layers
8layer_metrics
	variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object


,kernel
-recurrent_kernel
.bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+&call_and_return_all_conditional_losses
__call__"į
_tf_keras_layerĒ{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
¹
=layer_regularization_losses
regularization_losses
>non_trainable_variables
trainable_variables
?metrics

@layers
Alayer_metrics
	variables

Bstates
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Clayer_regularization_losses
regularization_losses
Dnon_trainable_variables
trainable_variables
Emetrics

Flayers
Glayer_metrics
	variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
: 2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Hlayer_regularization_losses
regularization_losses
Inon_trainable_variables
trainable_variables
Jmetrics

Klayers
Llayer_metrics
	variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 :62dense_1/kernel
:62dense_1/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
­
Mlayer_regularization_losses
#regularization_losses
Nnon_trainable_variables
$trainable_variables
Ometrics

Players
Qlayer_metrics
%	variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
%:#`2gru/gru_cell/kernel
/:- `2gru/gru_cell/recurrent_kernel
#:!`2gru/gru_cell/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
°
Tlayer_regularization_losses
9regularization_losses
Unon_trainable_variables
:trainable_variables
Vmetrics

Wlayers
Xlayer_metrics
;	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
»
	Ytotal
	Zcount
[	variables
\	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	]total
	^count
_
_fn_kwargs
`	variables
a	keras_api"æ
_tf_keras_metric¤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
Y0
Z1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
,:*	č2Adam/embedding/embeddings/m
#:! 2Adam/dense/kernel/m
:2Adam/dense/bias/m
%:#62Adam/dense_1/kernel/m
:62Adam/dense_1/bias/m
*:(`2Adam/gru/gru_cell/kernel/m
4:2 `2$Adam/gru/gru_cell/recurrent_kernel/m
(:&`2Adam/gru/gru_cell/bias/m
,:*	č2Adam/embedding/embeddings/v
#:! 2Adam/dense/kernel/v
:2Adam/dense/bias/v
%:#62Adam/dense_1/kernel/v
:62Adam/dense_1/bias/v
*:(`2Adam/gru/gru_cell/kernel/v
4:2 `2$Adam/gru/gru_cell/recurrent_kernel/v
(:&`2Adam/gru/gru_cell/bias/v
ā2ß
E__inference_sequential_layer_call_and_return_conditional_losses_14095
E__inference_sequential_layer_call_and_return_conditional_losses_14120
E__inference_sequential_layer_call_and_return_conditional_losses_15062
E__inference_sequential_layer_call_and_return_conditional_losses_14653Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ę2ć
 __inference__wrapped_model_11608¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *.¢+
)&
embedding_input’’’’’’’’’
ö2ó
*__inference_sequential_layer_call_fn_14167
*__inference_sequential_layer_call_fn_15104
*__inference_sequential_layer_call_fn_14213
*__inference_sequential_layer_call_fn_15083Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ī2ė
D__inference_embedding_layer_call_and_return_conditional_losses_15114¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_embedding_layer_call_fn_15121¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ū2Ų
>__inference_gru_layer_call_and_return_conditional_losses_15508
>__inference_gru_layer_call_and_return_conditional_losses_16304
>__inference_gru_layer_call_and_return_conditional_losses_16691
>__inference_gru_layer_call_and_return_conditional_losses_15895Õ
Ģ²Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ļ2ģ
#__inference_gru_layer_call_fn_15906
#__inference_gru_layer_call_fn_16702
#__inference_gru_layer_call_fn_15917
#__inference_gru_layer_call_fn_16713Õ
Ģ²Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ß2Ü
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_16719
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_16730Æ
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
©2¦
8__inference_global_average_pooling1d_layer_call_fn_16735
8__inference_global_average_pooling1d_layer_call_fn_16724Æ
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ź2ē
@__inference_dense_layer_call_and_return_conditional_losses_16746¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ļ2Ģ
%__inference_dense_layer_call_fn_16755¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ģ2é
B__inference_dense_1_layer_call_and_return_conditional_losses_16766¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ń2Ī
'__inference_dense_1_layer_call_fn_16775¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
:B8
#__inference_signature_wrapper_14244embedding_input
Ä2Į¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ä2Į¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 __inference__wrapped_model_11608w,-.!"8¢5
.¢+
)&
embedding_input’’’’’’’’’
Ŗ "1Ŗ.
,
dense_1!
dense_1’’’’’’’’’6¢
B__inference_dense_1_layer_call_and_return_conditional_losses_16766\!"/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’6
 z
'__inference_dense_1_layer_call_fn_16775O!"/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’6 
@__inference_dense_layer_call_and_return_conditional_losses_16746\/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "%¢"

0’’’’’’’’’
 x
%__inference_dense_layer_call_fn_16755O/¢,
%¢"
 
inputs’’’’’’’’’ 
Ŗ "’’’’’’’’’§
D__inference_embedding_layer_call_and_return_conditional_losses_15114_/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ ")¢&

0’’’’’’’’’
 
)__inference_embedding_layer_call_fn_15121R/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’·
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_16719`7¢4
-¢*
$!
inputs’’’’’’’’’ 

 
Ŗ "%¢"

0’’’’’’’’’ 
 Ņ
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_16730{I¢F
?¢<
63
inputs'’’’’’’’’’’’’’’’’’’’’’’’’’’’

 
Ŗ ".¢+
$!
0’’’’’’’’’’’’’’’’’’
 
8__inference_global_average_pooling1d_layer_call_fn_16724S7¢4
-¢*
$!
inputs’’’’’’’’’ 

 
Ŗ "’’’’’’’’’ Ŗ
8__inference_global_average_pooling1d_layer_call_fn_16735nI¢F
?¢<
63
inputs'’’’’’’’’’’’’’’’’’’’’’’’’’’’

 
Ŗ "!’’’’’’’’’’’’’’’’’’³
>__inference_gru_layer_call_and_return_conditional_losses_15508q,-.?¢<
5¢2
$!
inputs’’’’’’’’’

 
p

 
Ŗ ")¢&

0’’’’’’’’’ 
 ³
>__inference_gru_layer_call_and_return_conditional_losses_15895q,-.?¢<
5¢2
$!
inputs’’’’’’’’’

 
p 

 
Ŗ ")¢&

0’’’’’’’’’ 
 Ķ
>__inference_gru_layer_call_and_return_conditional_losses_16304,-.O¢L
E¢B
41
/,
inputs/0’’’’’’’’’’’’’’’’’’

 
p

 
Ŗ "2¢/
(%
0’’’’’’’’’’’’’’’’’’ 
 Ķ
>__inference_gru_layer_call_and_return_conditional_losses_16691,-.O¢L
E¢B
41
/,
inputs/0’’’’’’’’’’’’’’’’’’

 
p 

 
Ŗ "2¢/
(%
0’’’’’’’’’’’’’’’’’’ 
 
#__inference_gru_layer_call_fn_15906d,-.?¢<
5¢2
$!
inputs’’’’’’’’’

 
p

 
Ŗ "’’’’’’’’’ 
#__inference_gru_layer_call_fn_15917d,-.?¢<
5¢2
$!
inputs’’’’’’’’’

 
p 

 
Ŗ "’’’’’’’’’ ¤
#__inference_gru_layer_call_fn_16702},-.O¢L
E¢B
41
/,
inputs/0’’’’’’’’’’’’’’’’’’

 
p

 
Ŗ "%"’’’’’’’’’’’’’’’’’’ ¤
#__inference_gru_layer_call_fn_16713},-.O¢L
E¢B
41
/,
inputs/0’’’’’’’’’’’’’’’’’’

 
p 

 
Ŗ "%"’’’’’’’’’’’’’’’’’’ ¼
E__inference_sequential_layer_call_and_return_conditional_losses_14095s,-.!"@¢=
6¢3
)&
embedding_input’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’6
 ¼
E__inference_sequential_layer_call_and_return_conditional_losses_14120s,-.!"@¢=
6¢3
)&
embedding_input’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’6
 ³
E__inference_sequential_layer_call_and_return_conditional_losses_14653j,-.!"7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’6
 ³
E__inference_sequential_layer_call_and_return_conditional_losses_15062j,-.!"7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’6
 
*__inference_sequential_layer_call_fn_14167f,-.!"@¢=
6¢3
)&
embedding_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’6
*__inference_sequential_layer_call_fn_14213f,-.!"@¢=
6¢3
)&
embedding_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’6
*__inference_sequential_layer_call_fn_15083],-.!"7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’6
*__inference_sequential_layer_call_fn_15104],-.!"7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’6²
#__inference_signature_wrapper_14244,-.!"K¢H
¢ 
AŖ>
<
embedding_input)&
embedding_input’’’’’’’’’"1Ŗ.
,
dense_1!
dense_1’’’’’’’’’6