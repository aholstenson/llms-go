# Changelog

## [0.7.0](https://github.com/aholstenson/llms-go/compare/v0.6.1...v0.7.0) (2026-08-28)


### Features

* Update model metadata ([b629414](https://github.com/aholstenson/llms-go/commit/b6294149bc09415d2cd7af79eefdc99a83536ec5))

## [0.6.1](https://github.com/aholstenson/llms-go/compare/v0.6.0...v0.6.1) (2026-08-10)


### Bug Fixes

* Use adaptive thinking for Claude 5 family models ([05ce5f2](https://github.com/aholstenson/llms-go/commit/05ce5f224b012824cc2c2f15adc4683ba36c1e3f))

## [0.6.0](https://github.com/aholstenson/llms-go/compare/v0.5.0...v0.6.0) (2026-08-04)


### Features

* Support supplying custom credentials to models ([4a7cb98](https://github.com/aholstenson/llms-go/commit/4a7cb982742070dbd86b2d1cc486b904fc83c4f2))
* Update model metadata ([2360471](https://github.com/aholstenson/llms-go/commit/2360471e10132c42b4082d163b90f1060df6d22f))

## [0.5.0](https://github.com/aholstenson/llms-go/compare/v0.4.0...v0.5.0) (2026-06-24)


### Features

* Add session snapshot support ([4d06cbb](https://github.com/aholstenson/llms-go/commit/4d06cbbba50cb9ba5b4a3b54ec282826eecc689f))
* Support for returning images and binary data for tool calls ([154baac](https://github.com/aholstenson/llms-go/commit/154baac2be359621ac0a7f026958521f5af7ec60))

## [0.4.0](https://github.com/aholstenson/llms-go/compare/v0.3.0...v0.4.0) (2026-06-04)


### Features

* Account for partial usage when models error ([d8e4631](https://github.com/aholstenson/llms-go/commit/d8e4631a115bdd7768d4105e5504a3296c72e435))
* Improve error messages from models ([9fd0d86](https://github.com/aholstenson/llms-go/commit/9fd0d860918f820cfee67bdb80da098357e24d68))


### Bug Fixes

* Always use streaming mode for Anthropic to avoid refusals due to request size ([3d2a70c](https://github.com/aholstenson/llms-go/commit/3d2a70cfa35dc64f36a70e66c8a46430d001e128))
* **anthropic:** Fix edge case with tool result caching ([292f5d3](https://github.com/aholstenson/llms-go/commit/292f5d39eb20ccd5beff872a28362e26030cb45d))
* Don't leak stale Retry-After into UnavailableError ([47d7cc7](https://github.com/aholstenson/llms-go/commit/47d7cc70ee2e5e2d8534bb3cbf5226e3c966ec3b))
* **google:** Don't leak messages in error logs ([5953299](https://github.com/aholstenson/llms-go/commit/5953299cdf5e38e8078c01755ac35c1cda0f9820))
* **jsonstream:** Always flush sub-parser at end of streaming string ([e0bb494](https://github.com/aholstenson/llms-go/commit/e0bb4946d51b9de96906eedb3bd2c771d34f6fdb))
* **jsonstream:** Propagate errors from sub-parsers ([fa1abfa](https://github.com/aholstenson/llms-go/commit/fa1abfa684cae5a63629e559dcff5b1499ae31be))

## [0.3.0](https://github.com/aholstenson/llms-go/compare/v0.2.0...v0.3.0) (2026-05-22)


### Features

* Add WithReasoningEffort as the main way of controlling thinking ([9442bd8](https://github.com/aholstenson/llms-go/commit/9442bd87734496444960bfd939124b6747b6cfa6))

## [0.2.0](https://github.com/aholstenson/llms-go/compare/v0.1.0...v0.2.0) (2026-05-22)


### Features

* Add StreamingEventToolError event ([deed93a](https://github.com/aholstenson/llms-go/commit/deed93a43439f5f9fbdfac1257bb750e1fdd4c29))
* Improve accounting of cache write tokens ([84454c6](https://github.com/aholstenson/llms-go/commit/84454c639c0255fd1ddc98a173a1fd1ea150d66f))
* Unify how input vs cached tokens are calculated ([6115fb1](https://github.com/aholstenson/llms-go/commit/6115fb1487af0964f99268df6ef288cff7065d84))
