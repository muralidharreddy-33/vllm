# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools

from fastapi import Request
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    """Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.
    """

    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):

        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait([handler_task, cancellation_task],
                                           return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


class ConcurrentRequestsMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request, call_next):
        if not request.url.path.startswith("/v1"):
            return await call_next(request)

        with request.app.state.server_load_metrics_lock:
            request.app.state.server_load_metrics += 1

        async def decrement():
            with request.app.state.server_load_metrics_lock:
                request.app.state.server_load_metrics -= 1

        response = None
        exc = None
        try:
            response = await call_next(request)
        except Exception as e:
            exc = e
        finally:
            if response is None:
                await decrement()
            else:
                if response.background is None:
                    response.background = BackgroundTask(decrement)
                else:
                    # Chain decrement after the existing background task
                    response.background = BackgroundTask(
                        lambda: asyncio.create_task(decrement()))
        if exc:
            raise exc
        return response
