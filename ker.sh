nvidia-docker run --memory-swappiness=0 \
		  --oom-kill-disable \
		  --memory-swap="32g" \
		  -m 16g \
		  -v $(pwd)/data:/data \
	      -v $(pwd)/src:/src \
          -v $(pwd)/models:/models \
          -it ker $@