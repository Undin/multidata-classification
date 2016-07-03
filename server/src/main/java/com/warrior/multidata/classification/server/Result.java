package com.warrior.multidata.classification.server;

/**
 * Created by warrior on 03/07/16.
 */
public class Result {

    public final String id;
    public final String gender;
    public final String relationship;

    public Result(String id, String gender, String relationship) {
        this.id = id;
        this.gender = gender;
        this.relationship = relationship;
    }

    public static Result defaultResult(String id) {
        return new Result(id, "female", "single");
    }
}
