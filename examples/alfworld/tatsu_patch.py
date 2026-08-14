"""Monkey-patch tatsu 5.8.3 for Python 3.12 compatibility.

tatsu 5.8.3 has stack corruption bugs on Python 3.12 due to changes in
contextmanager/exception-group semantics. The _statestack and _rule_stack
get out of sync, causing IndexError on pop().

This patch wraps the problematic pop() calls to be safe (no-op on empty).
Must be imported before textworld.
"""
import sys

if sys.version_info >= (3, 12):
    try:
        import tatsu.contexts as _ctx

        _orig_call = _ctx.ParseContext._call

        def _safe_call(self, ruleinfo):
            self._rule_stack.append(ruleinfo.name)
            try:
                result = self._recursive_call(ruleinfo)
                return result
            except _ctx.FailedParse:
                raise
            finally:
                if self._rule_stack:
                    self._rule_stack.pop()

        # Patch state property to handle empty stack
        _orig_state_getter = _ctx.ParseContext.state.fget

        def _safe_state_getter(self):
            if not self._statestack:
                # Re-push a fresh state if stack was corrupted
                self._push_ast()
            return self._statestack[-1]

        _ctx.ParseContext.state = property(_safe_state_getter)

        # Patch _pop_ast to be safe
        _orig_pop_ast = _ctx.ParseContext._pop_ast

        def _safe_pop_ast(self):
            if self._statestack:
                self._statestack.pop()

        _ctx.ParseContext._pop_ast = _safe_pop_ast

        # Patch _pop_cst to be safe
        if hasattr(_ctx.ParseContext, '_pop_cst'):
            _orig_pop_cst = _ctx.ParseContext._pop_cst

            def _safe_pop_cst(self):
                if self._statestack:
                    return _orig_pop_cst(self)

            _ctx.ParseContext._pop_cst = _safe_pop_cst

        # Patch _call to handle empty _rule_stack
        def _patched_call(self, ruleinfo):
            try:
                return _orig_call(self, ruleinfo)
            except IndexError:
                # Stack corruption — return failed parse
                raise _ctx.FailedParse(
                    self._tokenizer,
                    getattr(ruleinfo, 'name', '?'),
                    None,
                )

        _ctx.ParseContext._call = _patched_call

    except (ImportError, AttributeError):
        pass
