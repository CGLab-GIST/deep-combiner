docker build -t combiner .
nvidia-docker run \
	--rm \
	-v ${PWD}/data:/data \
	-v ${PWD}/codes:/codes \
	-it combiner;
