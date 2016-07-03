package com.warrior.multidata.classification.server;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Created by warrior on 03/07/16.
 */
@RestController
public class ClassificationRestController {

    @RequestMapping("/classification")
    public Result classification(@RequestParam(name = "id") String id) {
        return Result.defaultResult(id);
    }
}
